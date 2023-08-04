import time, functools, torch, os, random, utils, fnmatch, psutil, argparse, math
from prettytable import PrettyTable
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
#import torch.multiprocessing as mp
#import wandb

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__()
        # Time information incorporated via Gaussian random feature encoding
        # Randomly sampled weights initialisation. Fixed during optimisation i.e. not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, time):
        # Multiply batch of times by network weights
        time_proj = time[:, None] * self.W[None, :] * 2 * np.pi
        # Output [sin(2pi*wt);cos(2pi*wt)]
        gauss_out = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)
        return gauss_out

class Dense(nn.Module):
    """Fully connected layer that reshapes output of embedded conditional variable to feature maps"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        """Dense nn layer output must have same dimensions as input data:
            [batchsize, (dummy)nhits, (dummy)features]
        """
        return self.dense(x)[..., None]

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout):
        """Encoder block:
        Args:
        embed_dim: length of embedding / dimension of the model
        num_heads: number of parallel attention heads to use
        hidden: dimensionaliy of hidden layer
        dropout: regularising layer
        """
        super().__init__()
        # batch_first=True because normally in NLP the batch dimension would be the second dimension
        # In everything(?) else it is the first dimension so this flag is set to true to match other conventions
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=embed_dim, vdim=embed_dim, batch_first=True, dropout=0)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.fc1_cls = nn.Linear(embed_dim, hidden)
        self.fc2_cls = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.act_dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self,x,x_cls,src_key_padding_mask=None,):
        # Stash original embedded input
        residual = x.clone()
        # Multiheaded self-attention but replacing query with a single mean field approximator
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        x_cls = self.act(self.fc1_cls(x_cls))
        x_cls = self.act_dropout(x_cls)
        x_cls = self.fc2(x_cls)
        # Add mean field approximation to input embedding (acts like a bias)
        x = x + x_cls.clone()
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        # Add to original input embedding
        x = x + residual
        return x

class transformerblock(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, src, src_key_padding_mask=None):
        _src = self.attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        _src = self.dropout1(_src)
        src = src + _src.clone()
        src = self.norm1( src )
        _src = self.ffnn(src)
        _src = self.dropout2(_src)
        src = src + _src.clone()
        src = self.norm2( src )
        return src

class Gen(nn.Module):
    def __init__(self, n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std, **kwargs):
        '''Transformer encoder model
        Arguments:
        n_feat_dim = number of features
        embed_dim = dimensionality to embed input
        hidden_dim = dimensionaliy of hidden layer
        num_encoder_blocks = number of encoder blocks
        num_attn_heads = number of parallel attention heads to use
        dropout_gen = regularising layer
        marginal_prob_std = standard deviation of Gaussian perturbation captured by SDE
        '''
        super().__init__()
        # Embedding: size of input (n_feat_dim) features -> size of output (embed_dim)
        self.embed = nn.Linear(n_feat_dim, embed_dim)
        # Seperate embedding for (time/incident energy) conditional inputs (small NN with fixed weights)
        self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        # Boils embedding down to single value
        self.dense1 = Dense(embed_dim, 1)
        # Module list of encoder blocks
        self.encoder = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_attn_heads,
                    hidden=hidden_dim,
                    dropout=dropout_gen,
                )
                #transformerblock(
                #    embed_dim=embed_dim,
                #    n_heads=num_attn_heads,
                #    dropout=dropout_gen,
                #    hidden_dim=hidden_dim,
                #)
                for i in range(num_encoder_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(embed_dim, n_feat_dim)
        # token simply has same dimension as input feature embedding
        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim), requires_grad=True)
        self.act = nn.GELU()
        # Swish activation function
        self.act_sig = lambda x: x * torch.sigmoid(x)
        # Standard deviation of SDE
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, e, mask=None):
        # Embed 4-vector input 
        x = self.embed(x)
        # Add time embedding
        embed_t_ = self.act_sig( self.embed_t(t) )
        # Now need to get dimensions right
        x += self.dense1(embed_t_).clone()
        # Add incident particle energy embedding
        embed_e_ = self.embed_t(e)
        embed_e_ = self.act_sig(embed_e_)
        # Now need to get dimensions right
        x += self.dense1(embed_e_).clone()
        # 'class' token (mean field)
        x_cls = self.cls_token.expand(x.size(0), 1, -1)
        # Feed input embeddings into encoder block
        for layer in self.encoder:
            # Each encoder block takes previous blocks output as input
            #x = layer(x, x_cls, mask)
            x = layer(x, x, mask)
            #x = layer(x, mask) # transformerencoder layers
        # Rescale models output (helps capture the normalisation of the true scores)
        mean_ , std_ = self.marginal_prob_std(x,t)
        output = self.out(x) / std_[:, None, None]
        return output

def loss_fn(model, x, incident_energies, marginal_prob_std , eps=1e-5, device='cpu'):
    """The loss function for training score-based generative models
    Uses the weighted sum of Denoising Score matching objectives
    Denoising score matching
    - Perturbs data points with pre-defined noise distribution
    - Uses score matching objective to estimate the score of the perturbed data distribution
    - Perturbation avoids need to calculate trace of Jacobian of model output

    Args:
        model: A PyTorch model instance that represents a time-dependent score-based model
        x: A mini-batch of training data
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
        eps: A tolerance value for numerical stability
    """
    # Generate padding mask for padded entries
    # If a BoolTensor is provided, positions with True are ignored while False values will be unchanged
    padding_mask = (x[:,:,0]==-20).type(torch.bool)
    # Inverse mask to ignore models output for 0-padded hits in loss
    output_mask = (x[:,:,0]!=-20).type(torch.int)
    output_mask = output_mask.unsqueeze(-1)
    output_mask = output_mask.expand(output_mask.size()[0], output_mask.size()[1],4)
    # Tensor of randomised conditional variable 'time' steps
    random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps
    # Tensor of conditional variable incident energies 
    incident_energies = torch.squeeze(incident_energies,-1)
    # Noise matrix
    # Multiply by mask so we don't go perturbing zero padded values to have some non-sentinel value
    z = torch.randn_like(x)*output_mask
    z = z.to(device)
    # Sample from standard deviation of noise
    mean_, std_ = marginal_prob_std(x,random_t)
    # Add noise to input
    perturbed_x = x + z * std_[:, None, None]
    # Evaluate model (aim: to estimate the score function of each noise-perturbed distribution)
    model_output = model(perturbed_x, random_t, incident_energies, mask=padding_mask)
    # Calculate loss 
    losses = (model_output*std_[:,None,None] + z)**2
    # Zero losses calculated over padded inputs
    losses = losses*output_mask
    # Sum loss across all hits and 4-vectors
    sum_loss = torch.sum( losses, dim=(1,2))
    # Average across batch
    batch_loss = torch.mean( sum_loss )
    
    return batch_loss

def  pc_sampler(score_model, marginal_prob_std, diffusion_coeff, sampled_e_h_, batch_size=1, snr=0.16, device='cuda', eps=1e-3):
    ''' Generate samples from score based models with Predictor-Corrector method
        Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the std of the perturbation kernel
        diffusion_coeff: A function that gives the diffusion coefficient 
        of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

        Returns:
            samples
    '''
    sampled_energies = torch.tensor(sampled_e_h_[0])
    sampled_hits = torch.tensor(sampled_e_h_[1])
    padded_hits = torch.tensor(sampled_e_h_[2])
    print('padded_hits: ', padded_hits)

    num_steps=100
    t = torch.ones(batch_size, device=device)
    gen_n_hits = int(sampled_hits.item())
    init_x = torch.randn(batch_size, gen_n_hits, 4, device=device)

    # Pad initial input in same way input to model looked during training
    pad_hits = padded_hits-sampled_hits
    padded_shower = F.pad(input = init_x, pad=(0,0,0,pad_hits), mode='constant', value=-20)

    # Mean is the only thing related to input hits, std only related to conditional
    mean_,std_ =  marginal_prob_std(padded_shower,t)
    std_.to(device)
    
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0]-time_steps[1]
    x = padded_shower*std_[:,None,None]
    print(f'Check padded shower for model: {x}')

    # Padding masks defined by initial # hits / zero padding
    padding_mask = (x[:,:,0]==-20).type(torch.bool)

    # Inverse mask to ignore models output for 0-padded hits in loss
    output_mask = (x[:,:,0]!=-20).type(torch.int)
    output_mask = output_mask.unsqueeze(-1)
    output_mask = output_mask.expand(output_mask.size()[0], output_mask.size()[1],4)

    with torch.no_grad():
         for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=x.device) * time_step
            
            # matrix multiplication in GaussianFourier projection doesnt like float64
            sampled_energies = sampled_energies.to(x.device, torch.float32)
            alpha = torch.ones_like(torch.tensor(time_step))
            
            # Corrector step (Langevin MCMC)
            # First calculate Langevin step size using the predicted scores
            grad = score_model(x, batch_time_step, sampled_energies, mask=padding_mask)
            print('grad: ', grad)

            # Noise to add to input (multiply by mask as we don't want to add noise to padded values)
            noise = torch.randn_like(x)
            noise = noise * output_mask
            
            # Vector norm (sqrt sum squares)
            # Take the mean value of the vector norm (sqrt sum squares), of the flattened scores for e,x,y,z
            flattened_scores = grad.reshape(grad.shape[0], -1)
            grad_norm = torch.linalg.norm( flattened_scores, dim=-1 ).mean()
            flattened_noise = noise.reshape(noise.shape[0],-1)
            noise_norm = torch.linalg.norm( flattened_noise, dim=-1 ).mean()
            langevin_step_size =  (snr * noise_norm / grad_norm)**2 * 2 * alpha
            
            # Implement iteration rule
            x_mean = x + langevin_step_size * grad
            x = x_mean + torch.sqrt(2 * langevin_step_size) * noise

            # Euler-Maruyama predictor step
            drift, diff = diffusion_coeff(x,batch_time_step)
            x_mean = x + (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies, mask=padding_mask) * step_size
            print('x_mean: ', x_mean.shape)
            print('torch.sqrt(diff**2 * step_size)[:, None, None]: ', torch.sqrt(diff**2 * step_size)[:, None, None].shape)
            x = x_mean + torch.sqrt(diff**2 * step_size)[:, None, None] * torch.randn_like(x)
            
    # Do not include noise in last step
    # remove padded hits
    print('final output: ', x_mean)
    x_mean = x_mean * output_mask
    return x_mean

def check_mem():
    # Resident set size memory (non-swap physical memory process has used)
    process = psutil.Process(os.getpid())
    print('Memory usage of current process 0 [MB]: ', process.memory_info().rss/1000000)
    return

def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-o','--output',dest='output_path', help='Path to output directory', default='', type=str)
    argparser.add_argument('-s','--switches',dest='switches', help='Binary representation of switches that run: evaluation plots, training, sampling, evaluation plots', default='0000', type=str)
    args = argparser.parse_args()
    workingdir = args.output_path
    switches_ = int('0b'+args.switches,2)
    switches_str = bin(int('0b'+args.switches,2))
    trigger = 0b0001
    print(f'switches trigger: {switches_str}')
    if switches_ & trigger:
        print('input_feature_plots = ON')
    if switches_>>1 & trigger:
        print('training_switch = ON')
    if switches_>>2 & trigger:
        print('sampling_switch = ON')
    if switches_>>3 & trigger:
        print('evaluation_plots_switch = ON')

    print('torch version: ', torch.__version__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device: ', device)
    if torch.cuda.is_available():
        print('Cuda used to build pyTorch: ',torch.version.cuda)
        print('Current device: ', torch.cuda.current_device())
        print('Cuda arch list: ', torch.cuda.get_arch_list())
    
    print('Working directory: ' , workingdir)
    
    # Useful when debugging gradient issues
    torch.autograd.set_detect_anomaly(True)

    ### SDE PARAMETERS ###
    SDE = 'VE'
    sigma_max = 50.
    ### MODEL PARAMETERS ###
    batch_size = 64
    lr = 0.00001
    n_epochs = 500
    n_feat_dim = 4
    embed_dim = 512
    hidden_dim = 128
    num_encoder_blocks = 6
    num_attn_heads = 8
    dropout_gen = 0
    # SAMPLE PARAMETERS
    n_sampler_calls = 100

    model_params_print = f'''
    ### PARAMS ###
    ### SDE PARAMETERS ###
    SDE = {SDE}
    sigma/beta max. = {sigma_max}
    ### MODEL PARAMETERS ###
    batch_size = {batch_size}
    lr = {lr}
    n_epochs = {n_epochs}
    n_feat_dim = {n_feat_dim}
    embed_dim = {embed_dim}
    hidden_dim = {hidden_dim}
    num_encoder_blocks = {num_encoder_blocks}
    num_attn_heads = {num_attn_heads}
    dropout_gen = {dropout_gen}
    # SAMPLE PARAMETERS
    n_sampler_calls = {n_sampler_calls}
    '''
    print(model_params_print)   
    
    # Instantiate stochastic differential equation
    if SDE == 'VP':
        sde = utils.VPSDE(beta_max=sigma_max,device=device)
    if SDE == 'VE':
        sde = utils.VESDE(sigma_max=sigma_max,device=device)
    marginal_prob_std_fn = functools.partial(sde.marginal_prob)
    diffusion_coeff_fn = functools.partial(sde.sde)

    # List of training input files
    training_file_path = '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/datasets/'
    files_list_ = []
    for filename in os.listdir(training_file_path):
        if fnmatch.fnmatch(filename, 'padded_dataset_1_photons_1_tensor_euclidian_0_rescaled_xy.pt'):
            files_list_.append(os.path.join(training_file_path,filename))

    #### Input plots ####
    if switches_ & trigger:
        files_list_ = files_list_[:1]
        for filename in files_list_:
            print('filename: ', filename)
            custom_data = utils.cloud_dataset(filename)
            #custom_data = utils.cloud_dataset(filename, transform=utils.rescale_energies(), transform_y=utils.rescale_conditional())
            #output_directory = workingdir+'/feature_plots_rescale_padded_dataset_1_photons_1_graph_0_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
            output_directory = workingdir+'/feature_plots_padded_dataset_1_photons_1_tensor_euclidian_0_rescaled_xy_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Load input data
            point_clouds_loader = DataLoader(custom_data,batch_size=1,shuffle=False)
            n_hits = []
            all_x = []
            all_y = []
            all_z = []
            all_e = []
            all_incident_e = []
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                # Limit number fo showers to plot for memories sake
                if i>500:
                    break
                # Resident set size memory (non-swap physical memory process has used)
                #process = psutil.Process(os.getpid())
                #print(f'Memory usage of current process shower {i} [MB]: {process.memory_info().rss/1000000}')
                
                hit_energies = shower_data[0][:,0]
                hit_xs = shower_data[0][:,1]
                hit_ys = shower_data[0][:,2]
                hit_zs = shower_data[0][:,3]

                mask = hit_energies.gt(-20.0)

                real_hit_energies = torch.masked_select(hit_energies,mask)
                real_hit_xs = torch.masked_select(hit_xs,mask)
                real_hit_ys = torch.masked_select(hit_ys,mask)
                real_hit_zs = torch.masked_select(hit_zs,mask)

                real_hit_energies = real_hit_energies.tolist()
                real_hit_xs = real_hit_xs.tolist()
                real_hit_ys = real_hit_ys.tolist()
                real_hit_zs = real_hit_zs.tolist()
                
                n_hits.append( len(real_hit_energies) )
                all_e.append(real_hit_energies)
                all_x.append(real_hit_xs)
                all_y.append(real_hit_ys)
                all_z.append(real_hit_zs)
                all_incident_e.append(incident_energies.item())

            plot_e = np.concatenate(all_e)
            plot_x = np.concatenate(all_x)
            plot_y = np.concatenate(all_y)
            plot_z = np.concatenate(all_z)
            
            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('# Hits')
            plt.hist(n_hits, 100, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'nhit.png')
            
            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit energy')
            plt.hist(plot_e, 100, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_energies.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit x position')
            plt.hist(plot_x, 100, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_x.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit y position')
            plt.hist(plot_y, 100, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_y.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit z position')
            plt.hist(plot_z, 50, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_z.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Incident energy')
            plt.hist(all_incident_e, 100, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_incident_e.png')

    #### Training ####
    if switches_>>1 & trigger:
        output_directory = workingdir+'/training_VESDE_beta50_nocls_6blocks_maskedloss_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        print('Output directory: ', output_directory)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        '''if tracking_ == True:
            if sweep_ == True:
                run_ = wandb.init()
                print('Running sweep!')
                # note that we define values from `wandb.config` instead of defining hard values
                lr  =  wandb.config.lr
                batch_size = wandb.config.batch_size
                n_epochs = wandb.config.epochs
            else:
                run_ = wandb.init()
                print('Tracking!')
                lr  =  wandb.config.lr
                batch_size = wandb.config.batch_size
                n_epochs = wandb.config.epochs'''
        
        model=Gen(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
        print('model: ', model)
        if torch.cuda.device_count() > 1:
            print(f'Lets use {torch.cuda.device_count()} GPUs!')
            model = nn.DataParallel(model)

        table = PrettyTable(['Module name', 'Parameters listed'])
        t_params = 0
        for name_ , para_ in model.named_parameters():
            if not para_.requires_grad: continue
            param = para_.numel()
            table.add_row([name_, param])
            t_params+=param
        print(table)
        print(f'Sum of trained parameters: {t_params}')

        # Optimiser needs to know model parameters for to optimise
        optimiser = Adam(model.parameters(),lr=lr)
        
        av_training_losses_per_epoch = []
        av_testing_losses_per_epoch = []
        for epoch in range(0,n_epochs):
            print(f"epoch: {epoch}")
            # Create/clear per epoch variables
            cumulative_epoch_loss = 0.
            cumulative_test_epoch_loss = 0.

            file_counter = 0
            n_training_showers = 0
            n_testing_showers = 0
            # Load files
            files_list_ = [files_list_[f] for f in range(0,1)]
            # For debugging purposes
            files_list_ = files_list_[:1]
            
            for filename in files_list_:
                file_counter+=1

                # Rescaling now done in padding package
                custom_data = utils.cloud_dataset(filename, device=device)
                print(f'{len(custom_data)} showers in file')
                
                train_size = int(0.9 * len(custom_data.data))
                test_size = len(custom_data.data) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])

                n_training_showers+=train_size
                n_testing_showers+=test_size
            
                # Load clouds for each epoch of data dataloaders length will be the number of batches
                shower_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                shower_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                print(f'Size of training dataset: {len(train_dataset)} showers, batch = {batch_size} showers')
                
                # Load a shower for training
                for i, (shower_data,incident_energies) in enumerate(shower_loader_train,0):
                    
                    #check_mem()
                    
                    # Move model to device and set dtype as same as data (note torch.double works on both CPU and GPU)
                    model.to(device, shower_data.dtype)
                    shower_data.to(device)
                    incident_energies.to(device)

                    if len(shower_data) < 1:
                        print('Very few hits in shower: ', len(shower_data))
                        continue
                    # Calculate loss average for batch
                    loss = loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, device=device)
                    cumulative_epoch_loss+=float(loss)
                    # Zero any gradients from previous steps
                    optimiser.zero_grad()
                    # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
                    loss.backward(retain_graph=True)
                    # Update value of x += -lr * x.grad
                    optimiser.step()
            
                # Testing on subset of file
                for i, (shower_data,incident_energies) in enumerate(shower_loader_test,0):
                    with torch.no_grad():
                        test_loss = loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, device=device)
                        cumulative_test_epoch_loss+=float(test_loss)
            
            #wandb.log({"training_loss": cumulative_epoch_loss/n_training_showers,
            #           "testing_loss": cumulative_test_epoch_loss/n_testing_showers})

            # Add the batch size just used to the total number of clouds
            av_training_losses_per_epoch.append(cumulative_epoch_loss/n_training_showers)
            av_testing_losses_per_epoch.append(cumulative_test_epoch_loss/n_testing_showers)
            print(f'End-of-epoch: average train loss = {av_training_losses_per_epoch}, average test loss = {av_testing_losses_per_epoch}')
            if epoch % 50 == 0:
                # Save checkpoint file after each epoch
                torch.save(model.state_dict(), output_directory+'ckpt_tmp_'+str(epoch)+'.pth')
        
        torch.save(model.state_dict(), output_directory+'ckpt_tmp_'+str(epoch)+'.pth')
        av_training_losses_per_epoch = av_training_losses_per_epoch
        av_testing_losses_per_epoch = av_testing_losses_per_epoch
        print('Training losses : ', av_training_losses_per_epoch)
        print('Testing losses : ', av_testing_losses_per_epoch)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.plot(av_training_losses_per_epoch, label='training')
        plt.plot(av_testing_losses_per_epoch, label='testing')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.savefig(output_directory+'loss_v_epoch.png')
        #wandb.finish()
    
    #### Sampling ####
    if switches_>>2 & trigger:    
        output_directory = workingdir+'/sampling_VPSDE_beta20_nocls_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        batch_size = 1
        model=Gen(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
        load_name = os.path.join(workingdir,'training_VPSDE_beta20_nocls_20230512_1329_output/ckpt_tmp_499.pth')

        model.load_state_dict(torch.load(load_name, map_location=device))
        model.to(device)
        samples_ = []
        in_energies = []
        total_hits_lengths = []
        real_hits_lengths = []
        sampled_energies = []
        # Use clouds from a sample of random files to generate a distribution of # hits
        for idx_ in random.sample( range(0,len(files_list_)), 1):
            custom_data = utils.cloud_dataset(files_list_[idx_], device=device)
            point_clouds_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=True)
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                # Get nhits for examples in input file
                total_hits = shower_data[0]
                mask = total_hits[:,0].gt(-20.0)
                real_hits = torch.masked_select(total_hits[:,0],mask)
                real_hits_lengths.append( len(real_hits) )
                total_hits_lengths.append( len(total_hits) )
                # Get incident energies from input file
                sampled_energies.append( incident_energies.tolist() )
        
        # Stack
        e_h_comb = list(zip(sampled_energies, real_hits_lengths, total_hits_lengths))
        for s_ in range(0,n_sampler_calls):
            # Generate random number to sample an example energy and nhits
            idx = np.random.randint(len(e_h_comb), size=1)[0]
            # Energies and hits to pass to sampler
            sampled_e_h_ = e_h_comb[idx]
            in_energies.append(sampled_e_h_[0])
            # Initiate sampler
            sampler = pc_sampler
            # Generate a sample of point clouds
            print(f'Generating point cloud: {s_}')
            samples = sampler(model, marginal_prob_std_fn, diffusion_coeff_fn, sampled_e_h_, batch_size, device=device)
            samples_.append(samples)
        torch.save([samples_,torch.as_tensor(in_energies)], output_directory+'generated_samples.pt')

    #### Evaluation plots ####
    if switches_>>3 & trigger:
        output_directory = workingdir+'/evaluation_regular_encoder_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        batch_size=1
        # Initialise clouds with detector structure
        layer_list = []
        xbins_list = []
        ybins_list = []
        for idx_ in random.sample(range(0,len(files_list_)),1):
            # Load input data (just need example file for now)
            print(f'GEANT file: {files_list_[idx_]}')
            custom_data = utils.cloud_dataset(files_list_[idx_],device=device)
            point_clouds_loader = DataLoader(custom_data,batch_size=batch_size,shuffle=False)
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                if i>100:
                    continue
                zlayers_ = shower_data[:,:,3].tolist()[0]
                xbins_ = shower_data[:,:,1].tolist()[0]
                ybins_ = shower_data[:,:,2].tolist()[0]
                for z in zlayers_:
                    layer_list.append( z )
                for x in xbins_:
                    xbins_list.append( x )
                for y in ybins_:
                    ybins_list.append( y )
        
        layer_set = set([z for z in layer_list])
        print('layer_set: ', layer_set)

        for idx_ in random.sample(range(0,len(files_list_)),1):

            n_hits_GEANT=[]
            all_e_GEANT=[]
            all_x_GEANT=[]
            all_y_GEANT=[]
            all_z_GEANT=[]
            all_incident_e_GEANT=[]
            total_shower_e_GEANT = []

            total_shower_e_per_layer_GEANT = {}
            total_shower_nhits_per_layer_GEANT = {}
            for layer_ in range(0,len(layer_set)):
                total_shower_e_per_layer_GEANT[layer_] = []
                total_shower_nhits_per_layer_GEANT[layer_] = []
            
            custom_data = utils.cloud_dataset(files_list_[idx_], device=device)
            point_clouds_loader = DataLoader(custom_data,batch_size=1,shuffle=False)
            
            # Load each cloud and calculate desired quantity
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                if i>100:
                    break

                hit_energies_GEANT = shower_data[0][:,0]
                hit_xs_GEANT = shower_data[0][:,1]
                hit_ys_GEANT = shower_data[0][:,2]
                hit_zs_GEANT = shower_data[0][:,3]
                mask = (hit_energies_GEANT != -20.0)

                real_hit_energies_GEANT = torch.masked_select(hit_energies_GEANT,mask)
                real_hit_xs_GEANT = torch.masked_select(hit_xs_GEANT,mask)
                real_hit_ys_GEANT = torch.masked_select(hit_ys_GEANT,mask)
                real_hit_zs_GEANT = torch.masked_select(hit_zs_GEANT,mask)

                real_hit_energies_GEANT = real_hit_energies_GEANT.tolist()
                real_hit_xs_GEANT = real_hit_xs_GEANT.tolist()
                real_hit_ys_GEANT = real_hit_ys_GEANT.tolist()
                real_hit_zs_GEANT = real_hit_zs_GEANT.tolist()
                
                n_hits_GEANT.append( len(real_hit_energies_GEANT) )
                all_e_GEANT.append(real_hit_energies_GEANT)
                all_x_GEANT.append(real_hit_xs_GEANT)
                all_y_GEANT.append(real_hit_ys_GEANT)
                all_z_GEANT.append(real_hit_zs_GEANT)
                all_incident_e_GEANT.append(incident_energies.item())
                total_shower_e_GEANT.append(sum(real_hit_energies_GEANT))
                
                # append sum of each clouds deposited energy/hits per layer to lists
                '''for layer_ in cloud_features.layer_set:
                    input_layers_sum_e[layer_].append(cloud_features.layers_sum_e.get(layer_)[0])
                    input_layers_nhits[layer_].append(cloud_features.layers_nhits.get(layer_)[0])'''
        
        # Load generated image file
        test_ge_filename = 'sampling_regular_encoder_20230505_1519_output/generated_samples.pt'
        
        custom_gendata = utils.cloud_dataset(test_ge_filename, device=device)
        gen_point_shower_loader = DataLoader(custom_gendata,batch_size=1,shuffle=False)

        n_hits_GEN=[]
        all_e_GEN=[]
        all_x_GEN=[]
        all_y_GEN=[]
        all_z_GEN=[]
        all_incident_e_GEN=[]
        total_shower_e_GEN = []

        total_shower_e_per_layer_GEN = {}
        total_shower_nhits_per_layer_GEN = {}       
        
        for layer_ in range(0,len(layer_set)):
                total_shower_e_per_layer_GEN[layer_] = []
                total_shower_nhits_per_layer_GEN[layer_] = []
        
        for i, (shower_data,incident_energies) in enumerate(gen_point_shower_loader,0):
            if i>100:
                    break
            
            shower_data = torch.squeeze(shower_data)
            hit_energies_GEN = shower_data[:,0]
            hit_xs_GEN = shower_data[:,1]
            hit_ys_GEN = shower_data[:,2]
            hit_zs_GEN = shower_data[:,3]
            
            mask = (hit_energies_GEN != 0.0)
            
            real_hit_energies_GEN = torch.masked_select(hit_energies_GEN,mask)
            
            if sum(real_hit_energies_GEN) > 0:
                print(f'Strange to have +ve sum of hits')
                continue
            
            real_hit_xs_GEN = torch.masked_select(hit_xs_GEN,mask)
            real_hit_ys_GEN = torch.masked_select(hit_ys_GEN,mask)
            real_hit_zs_GEN = torch.masked_select(hit_zs_GEN,mask)

            real_hit_energies_GEN = real_hit_energies_GEN.tolist()
            real_hit_xs_GEN = real_hit_xs_GEN.tolist()
            real_hit_ys_GEN = real_hit_ys_GEN.tolist()
            real_hit_zs_GEN = real_hit_zs_GEN.tolist()
            
            n_hits_GEN.append( len(real_hit_energies_GEN) )
            all_e_GEN.append(real_hit_energies_GEN)
            all_x_GEN.append(real_hit_xs_GEN)
            all_y_GEN.append(real_hit_ys_GEN)
            all_z_GEN.append(real_hit_zs_GEN)
            all_incident_e_GEN.append(incident_energies.item())
            total_shower_e_GEN.append(sum(real_hit_energies_GEN))
            
            '''
            # append each clouds deposited energy per layer to lists
            for layer_ in gen_cloud_features.layer_set:
                gen_layers_sum_e[layer_].append(gen_cloud_features.layers_sum_e.get(layer_)[0])
                gen_layers_nhits[layer_].append(gen_cloud_features.layers_nhits.get(layer_)[0])
            '''

        plot_e_GEANT = np.concatenate(all_e_GEANT)
        plot_x_GEANT = np.concatenate(all_x_GEANT)
        plot_y_GEANT = np.concatenate(all_y_GEANT)
        plot_z_GEANT = np.concatenate(all_z_GEANT)
        
        plot_e_GEN = np.concatenate(all_e_GEN)
        plot_x_GEN = np.concatenate(all_x_GEN)
        plot_y_GEN = np.concatenate(all_y_GEN)
        plot_z_GEN = np.concatenate(all_z_GEN)

        
        bins_nhit = list(range(0,np.max(n_hits_GEANT),20))
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('# entries')
        plt.xlabel('# Hits')
        plt.hist(n_hits_GEANT, bins=bins_nhit, color='purple', alpha=0.5, label='Geant4')
        plt.hist(n_hits_GEN, bins=bins_nhit, color='green', alpha=0.5, label='Gen')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'nhit.png')
        
        bins_e = np.histogram_bin_edges(plot_e_GEANT, bins='auto')
        bins_e_GEN = np.histogram_bin_edges(plot_e_GEN, bins='auto')
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('# entries')
        plt.xlabel('Hit energy')
        plt.xlim([-17.5,2.5])
        plt.yscale('log')
        plt.hist(plot_e_GEANT, bins=bins_e, color='purple', alpha=0.5, label='Geant4')
        plt.hist(plot_e_GEN, bins=bins_e_GEN, color='green', alpha=0.5, label='Gen')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'hit_energies.png')

        bins_x = np.histogram_bin_edges(plot_x_GEANT, bins='auto')
        bins_x_GEN = np.histogram_bin_edges(plot_x_GEN, bins='auto')
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('# entries')
        plt.xlabel('Hit x position')
        plt.hist(plot_x_GEANT, bins=bins_x, color='purple', alpha=0.5, label='Geant4')
        plt.hist(plot_x_GEN, bins=bins_x_GEN, color='green', alpha=0.5, label='Gen')
        plt.xlim([-0.3,0.3])
        plt.ylim([0,1300])
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'hit_x.png')

        bins_y = np.histogram_bin_edges(plot_y_GEANT, bins='auto')
        bins_y_GEN = np.histogram_bin_edges(plot_y_GEN, bins='auto')
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('# entries')
        plt.xlabel('Hit y position')
        plt.hist(plot_y_GEANT, bins=bins_y, color='purple', alpha=0.5, label='Geant4')
        plt.hist(plot_y_GEN, bins=bins_y_GEN, color='green', alpha=0.5, label='Gen')
        plt.xlim([-0.3,0.3])
        plt.ylim([0,1300])
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'hit_y.png')

        bins_z = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('# entries')
        plt.xlabel('Hit z position')
        plt.hist(plot_z_GEANT, bins=bins_z, color='purple', alpha=0.5, label='Geant4')
        plt.hist(plot_z_GEN, bins=bins_z, color='green', alpha=0.5, label='Gen')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'hit_z.png')

        bins_ine = np.arange(0,1,0.01)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('# entries')
        plt.xlabel('Incident energy')
        plt.hist(all_incident_e_GEANT, bins=bins_ine, color='purple', alpha=0.5, label='Geant4')
        plt.hist(all_incident_e_GEN, bins=bins_ine, color='green', alpha=0.5, label='Gen')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'hit_incident_e.png')


if __name__=='__main__':
    start = time.time()
    '''global tracking_
    global sweep_
    tracking_ = False
    sweep_ = False
    # Start sweep job.
    if tracking_:
            if sweep_:
                # Define sweep config
                sweep_configuration = {
                    'method': 'random',
                    'name': 'sweep',
                    'metric': {'goal': 'maximize', 'name': 'val_acc'},
                    'parameters': 
                    {
                        'batch_size': {'values': [50, 100, 150]},
                        'epochs': {'values': [5, 10, 15]},
                        'lr': {'max': 0.001, 'min': 0.00001},
                    }
                }
                sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
                wandb.agent(sweep_id, function=main, count=4)
            else:
                # start a new wandb run to track this script
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="trans_tdsm",
                    # track hyperparameters and run metadata
                    config={
                    "architecture": "encoder",
                    "dataset": "calochallenge_2",
                    "batch_size": 10,
                    "epochs": 10,
                    "lr": 0.0001,
                    }
                )
                main()
    else:
        main()'''
    main()
    fin = time.time()
    elapsed_time = fin-start
    print('Time elapsed: {:3f}'.format(elapsed_time))