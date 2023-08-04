import time, functools, torch, os, sys, random, utils, fnmatch, psutil, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import util.dataset_structure, util.display, util.model
import tqdm

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__() # inherits from pytorch nn class
        # Randomly sampled weights initialisation. Fixed during optimisation i.e. not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        print(f'Initial GaussianFourierProjection W weights: {self.W}')
    def forward(self, x):
        # Time information incorporated via Gaussian random feature encoding
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """Fully connected layer that reshapes outputs to feature maps"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        print(f'Initial Dense weights: {self.dense.weight}')
    def forward(self, x):
        """Dense nn layer output must have same dimensions as input data:
            For point clouds: [batchsize, (dummy)nhits, (dummy)features]
        """
        return self.dense(x)[..., None]

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout):
        """Initialise an encoder block:
        For 'class token' input (initially random):
            - Attention layer
            - Linear layer
            - GeLU activation
            - Linear layer
        
        For input:
            - Add class token block output
            - GeLU activation
            - Dropout regularisation layer
            - Linear layer
            - Add to original input

        Args:
        embed_dim: length of embedding
        num_heads: number of parallel attention heads to use
        hidden: dimensionaliy of hidden layer
        dropout: regularising layer
        """
        super().__init__()
        # batch_first=True because normally in NLP the batch dimension would be the second dimension
        # In everything(?) else it is the first dimension so this flag is set to true to match other conventions
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(embed_dim, hidden)
        
        self.fc2 = nn.Linear(hidden, embed_dim)
        
        self.fc1_cls = nn.Linear(embed_dim, hidden)
        
        self.fc2_cls = nn.Linear(hidden, embed_dim)
        
        self.act = nn.GELU()
        
        self.act_dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self,x,x_cls,src_key_padding_mask=None,):
        # Stash original embedded input
        residual = x.clone()

        # Multiheaded self-attention but replacing queries of all input examples with a single mean field approximator
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

class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, dropout_gen, marginal_prob_std, **kwargs):
        '''Transformer encoder model
        Arguments:
        n_dim = number of features
        l_dim_gen = dimensionality to embed input
        hidden_gen = dimensionaliy of hidden layer
        num_layers_gen = number of encoder blocks
        heads_gen = number of parallel attention heads to use
        dropout_gen = regularising layer
        marginal_prob_std = standard deviation of Gaussian perturbation captured by SDE
        '''
        super().__init__()

        # Embedding: size of input (n_dim) features -> size of output (l_dim_gen)
        self.embed = nn.Linear(n_dim, l_dim_gen)
        
        # Seperate embedding for (time/incident energy) conditional inputs (small NN with fixed weights)
        self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=64), nn.Linear(64, 64))
        # Boils embedding down to single value
        self.dense1 = Dense(64, 1)

        # Module list of encoder blocks
        self.encoder = nn.ModuleList(
            [
                Block(
                    embed_dim=l_dim_gen,
                    num_heads=heads_gen,
                    hidden=hidden_gen,
                    dropout=dropout_gen,
                )
                for i in range(num_layers_gen)
            ]
        )
        
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(l_dim_gen, n_dim)

        # token simply has same dimension as input feature embedding
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim_gen), requires_grad=True)
        self.cls_token = nn.Parameter(torch.ones(1, 1, l_dim_gen), requires_grad=True)
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
            x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)
            # Should concatenate with the time embedding after each block?

        mean_ , std_ = self.marginal_prob_std(x,t)
        std_1d = std_[:, None, None]
        output = self.out(mean_) / std_1d
        return output 

def loss_fn(model, x, incident_energies, marginal_prob_std , eps=1e-5, device='cpu',mask=None):
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
    process = psutil.Process(os.getpid())
    
    model.to(device)
    
    # Tensor of randomised conditional variable 'time' steps
    random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps
    
    # Tensor of conditional variable incident energies 
    incident_energies = torch.squeeze(incident_energies,-1)
    incident_energies.to(device)
    
    # matrix of noise
    z = torch.randn_like(x, device=device)
    
    # Sample from standard deviation of noise
    mean_, std_ = marginal_prob_std(x,random_t)
    std_.to(device)
    
    # Add noise to input
    #print(f'x.is_cuda: {x.is_cuda}')
    perturbed_x = x + z * std_[:, None, None]
    # Evaluate model
    model_output = model(perturbed_x, random_t, incident_energies, mask)
    
    losses = (model_output*std_[:,None,None] + z)**2
    if not mask is None:
      anti_mask = (~mask).float()
      anti_mask = anti_mask[...,None]
      losses = losses*anti_mask
    
    # Collect loss
    batch_loss = torch.sum( losses, dim=(0,1,2))
#    n_hits     = torch.sum( anti_mask, dim=(0,1)) if not mask is None else 1.0
    #cloud_loss = torch.mean( losses, dim=(0,1,2))
    #print(f'Summed losses (dim=1,2): {summed_losses}')
    return batch_loss

def  pc_sampler(score_model, marginal_prob_std, diffusion_coeff, sampled_energies, init_x, batch_size=1, snr=0.16, device='cuda', eps=1e-3, mask=True, padding_value=-20, jupyternotebook=False):
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
    
    num_steps=100
    t = torch.ones(batch_size, device=device)
    mean_,std_ =  marginal_prob_std(init_x,t)
    std_.to(device)
    mask_tensor = init_x[:,:,3] == padding_value
    init_x = init_x*std_[:,None,None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0]-time_steps[1]

    if jupyternotebook:
      time_steps = tqdm.notebook.tqdm(time_steps)
    x = init_x
#    print(f'input x shape: {init_x.shape}')
    with torch.no_grad():
         for time_step in time_steps:
            if not jupyternotebook:
              print(f"Sampler step: {time_step:.4f}")
            batch_time_step = torch.ones(batch_size, device=init_x.device) * time_step
            
            # matrix multiplication in GaussianFourier projection doesnt like float64
            sampled_energies = sampled_energies.to(x.device, torch.float32)
            alpha = torch.ones_like(torch.tensor(time_step))
            
            #for _ in range(1):
            # Corrector step (Langevin MCMC)
            # First calculate Langevin step size using the predicted scores
            if mask:
              grad = score_model(x, batch_time_step, sampled_energies, mask=mask_tensor)
            else:
              grad = score_model(x, batch_time_step, sampled_energies)

            #noise = np.prod(x.shape[1:])
            noise = torch.randn_like(x)
            
            # Vector norm (sqrt sum squares)
            # Take the mean value
            # of the vector norm (sqrt sum squares)
            # of the flattened scores for e,x,y,z
            if mask:
              mask_matrix = mask_tensor
              anti_mask = (~mask_matrix).float()
              anti_mask = anti_mask[...,None]
              grad  = grad*anti_mask
              noise = noise*anti_mask
 
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
            if mask:
              x_mean = x + (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies, mask=mask_tensor) * step_size
            else:
              x_mean = x + (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies) * step_size
            x = x_mean + torch.sqrt(diff**2 * step_size)[:, None, None] * torch.randn_like(x)
            
  #  print(f'x_mean: {x_mean}')
    # Do not include noise in last step
    return x_mean

def training(batch_size = 150,
             lr = 1e-4, 
             n_epochs = 200,
             model=None,
             new_marginal_prob_std_fn=None,
             device='cpu', 
             jupyternotebook=False,
             files_list=None,
             train_ratio = 0.9,
             mask=True,
             padding_value=-20,
             transform = None,
             transform_y = None,
             label=''):
    output_directory = 'training_result/' + label + '/'
    print('Output directory: ', output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    if model is None or files_list is None:
        print("Do not accept null model or empty file_list")
    if torch.cuda.device_count() > 1:
        print(f'Let us: {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)
    
    model.to(device)
    optimiser = Adam(model.parameters(), lr=lr)
    
    av_training_losses_per_epoch = []
    av_testing_losses_per_epoch = []
    
    fig, ax = plt.subplots(ncols=1, figsize=(4,4))
    
    if jupyternotebook:
        epochs = tqdm.notebook.trange(n_epochs)
        from IPython import display
        dh = display.display(fig, display_id=True)
    else:
        epochs = range(0, n_epochs)
    
    for epoch in epochs:
        
        if not jupyternotebook:
            print(f"epoch: {epoch}")
            
        cumulative_epoch_loss = 0.
        cumulative_test_epoch_loss = 0.
        
        file_counter = 0
        n_training_showers = 0
        n_testing_showers  = 0
        
        for file_name in files_list:
            file_counter += 1
            
            custom_data = util.dataset_structure.cloud_dataset(file_name, device=device, transform=transform,transform_y=transform_y)
            
#            print(f'{len(custom_data)} showers in file')
            
            train_size = int(0.9 * len(custom_data.data))
            test_size  = len(custom_data.data) - train_size
            

            train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])

            n_training_showers += train_size
            n_testing_showers  += test_size
            shower_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #TODO: pin_memory, num_workers have bug.
            shower_loader_test  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
#            print(f'train_dataset: {len(train_dataset)} showers, batch size: {len(shower_loader_train.dataset)} showers')
            for i,(shower_data, incident_energies)in enumerate(shower_loader_train,0):
                if len(shower_data) < 1:
                    print('Very few hits in shower: ', len(shower_data))
                    continue
    
                if mask:
                    loss = util.model.loss_fn(model, shower_data, incident_energies, new_marginal_prob_std_fn, device=device, mask=shower_data[:,:,3] == padding_value)
                else:
                    loss = util.model.loss_fn(model, shower_data, incident_energies, new_marginal_prob_std_fn, device=device)
                batch_loss_averaged = loss/len(shower_data)
                cumulative_epoch_loss += batch_loss_averaged.item()*batch_size
                optimiser.zero_grad()
                batch_loss_averaged.backward(retain_graph=True)
                optimiser.step()
                
            for i, (shower_data, incident_energies) in enumerate(shower_loader_test,0):
                with torch.no_grad():
                    if mask:
                        test_loss = util.model.loss_fn(model, shower_data, incident_energies, new_marginal_prob_std_fn, device=device, mask=shower_data[:,:,3]==padding_value)
                    else:
                        test_loss = util.model.loss_fn(model, shower_data, incident_energies, new_marginal_prob_std_fn, device=device)
                    test_batch_loss_averaged = test_loss/len(shower_data)
                    cumulative_test_epoch_loss+= test_batch_loss_averaged.item()*batch_size
            av_training_losses_per_epoch.append(cumulative_epoch_loss/n_training_showers)
            av_testing_losses_per_epoch.append(cumulative_test_epoch_loss/n_testing_showers)
            #print(f'End-of-epoch: average train loss = {av_training_losses_per_epoch}, average test loss = {av_testing_losses_per_epoch}')
            
            torch.save(model.state_dict(), output_directory+'ckpt_tmp_'+str(epoch)+'.pth')
        
        if not jupyternotebook:
            print('Training losses : ', av_training_losses_per_epoch)
            print('Testing losses : ', av_testing_losses_per_epoch)
        else:
            epochs.set_description('Average Loss: {:5f}(Train) {:5f}(Test)'.format(cumulative_epoch_loss/n_training_showers, cumulative_test_epoch_loss/n_testing_showers))

            fig, ax = plt.subplots(ncols=1, figsize=(4,4))
            plt.title('')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.yscale('log')
            plt.plot(av_training_losses_per_epoch, label='training')
            plt.plot(av_testing_losses_per_epoch, label='testing')
            plt.legend(loc='upper right')
            dh.update(fig)
            plt.close(fig)
            
    return model



## Sampling
def random_sampler(pdf,xbin):
    myCDF = np.zeros_like(xbin,dtype=float)
    myCDF[1:] = np.cumsum(pdf)
    a = np.random.uniform(0, 1)
    return xbin[np.argmax(myCDF>=a)-1]

def get_prob_dist(x,y,nbins):
    hist,xbin,ybin = np.histogram2d(x,y,bins=nbins)
    sum_ = hist.sum(axis=-1)
    sum_ = sum_[:,None]
    hist = hist/sum_
    hist[np.isnan(hist)] = 0.0
    return hist, xbin, ybin
def generate_hits(prob, xbin, ybin, x_vals, max_hits, n_features, device='cpu'):
    ind = np.digitize(x_vals, xbin) - 1
    ind[ind==len(xbin)-1] = len(xbin)-2
    ind[ind==-1] = 0    
    y_pred = []
    pred_nhits = []
    prob_ = prob[ind,:]
    for i in range(len(prob_)):
        nhits = int(random_sampler(prob_[i],ybin + 1))
        pred_nhits.append(nhits)
        y_pred.append(torch.randn(nhits, n_features, device=device))
    
    return pred_nhits, y_pred

def generate_sample(model=None,
                    marginal_prob_std=None,
                    diffusion_coeff=None,
                    sample_batch_size=150,
                    n_bin=100,
                    load_name = None,
                    device='cpu',
                    in_energies=None,
                    sampled_file_list=[],
                    mask=True,
                    padding_value = -20,
                    jupyternotebook = False,
                    label='sample',
                    transform   = None,
                    transform_y = None):
    
    in_energies = in_energies.cpu().numpy()
    
    output_directory = os.path.join('sampling', label)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("output_directory:", output_directory)
    model.load_state_dict(torch.load(load_name, map_location=device))
    model.to(device)
    
    # Generate 2D pdf from the training file
    entries = []
    all_incident_e = []
    max_hits = -1
    for file in sampled_file_list:
        custom_data = util.dataset_structure.cloud_dataset(file, device=device)
        point_clouds_loader = DataLoader(custom_data, batch_size=sample_batch_size, shuffle=False)
        
        for i, (shower_data, incident_energies) in enumerate(point_clouds_loader,0):
            valid_event = []
            data_np = shower_data.cpu().numpy().copy()
            energy_np = incident_energies.cpu().numpy().copy()
    
            masking = data_np[:,:,3] > -10

            for j in range(len(data_np)):
                valid_event = data_np[j][masking[j]]
                entries.append(len(valid_event))
                if len(valid_event)>max_hits:
                    max_hits = len(valid_event)
                all_incident_e.append(energy_np[j]) 
        del custom_data
        
    entries = np.array(entries)
    all_incident_e = np.array(all_incident_e)
    e_vs_nhits_prob, x_bin, y_bin = get_prob_dist(all_incident_e, entries, n_bin)    
    nhits, gen_hits = generate_hits(e_vs_nhits_prob, x_bin, y_bin, in_energies, max_hits, 4, device=device)
    torch.save([gen_hits, in_energies],'tmp.pt')
  
    gen_hits = util.dataset_structure.cloud_dataset('tmp.pt', device=device, transform=transform, transform_y=transform_y)
    gen_hits.padding(padding_value)
    os.system("rm tmp.pt")
    sample = []
    gen_hits_loader = DataLoader(gen_hits, batch_size=sample_batch_size, shuffle=False)
    for i, (gen_hit, sampled_energies) in enumerate(gen_hits_loader,0):
        sys.stdout.write('\r')
        sys.stdout.write("Progress: %d/%d" % ((i+1), len(gen_hits_loader)))
        sys.stdout.flush()
        generative = util.model.pc_sampler(model, marginal_prob_std, diffusion_coeff, sampled_energies, gen_hit, batch_size=len(gen_hit), snr=0.16, device=device, eps=1e-3, mask=mask, jupyternotebook=jupyternotebook, padding_value=padding_value)
        if i == 0:
            sample = generative
        else:
            sample = torch.cat([sample,generative])
    sample_ = []
    sample_np = sample.cpu().numpy()
    for i in range(len(sample_np)):
        tmp_sample = sample_np[i][:nhits[i]]
        sample_.append(torch.tensor(tmp_sample))
    torch.save([sample_,in_energies], os.path.join(output_directory, 'sample.pt'))
    return sample_, in_energies, nhits

