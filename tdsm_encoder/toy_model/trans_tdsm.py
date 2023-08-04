import time, functools, torch, os, random, utils, fnmatch, psutil, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#from CloudFeatures import CloudFeatures

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps"""
  def __init__(self, embed_dim, scale=30):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:,None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
  '''
  Fully connected layer that reshapes outputs to feature maps.
  '''
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None]

class Transformer(nn.Module):
  def __init__(self, embed_dim, num_heads, hidden, dropout):
    '''
    Initialise an encoder block:
    For 'class token' input (initially random):
        - Attention layer
        - Linear layer
        - GeLU activation
        - Linear layer
    For inut:
        - Add class token block output
        - GeLU activation
        - Dropout regularisation layer
        - Linear layer
        - Add to original input
    Args:
    embed_dim: length of embedding
    num_heads: number of parallel attention heads to use
    hidden: dimensionality of hidden layer
    dropout: regularising layer
    '''

    super().__init__()
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0)
    self.dropout = nn.Dropout(dropout)
    self.fc1  = nn.Linear(embed_dim, hidden)
    self.fc2  = nn.Linear(hidden, embed_dim)
    self.fc1_cls = nn.Linear(embed_dim, hidden)
    self.fc2_cls = nn.Linear(hidden, embed_dim)
    self.act = nn.GELU()
    self.act_dropout = nn.Dropout(dropout)
    self.hidden = hidden

  def forward(self, x, x_cls, src_key_padding_mask=None,):
    '''
    Args:
    x: shape(batch_size, nhit, nfeature)
    x_cls: shape(batch_size, 1, nfeature)
    '''
    residual = x.clone()
    x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0] # Q: x_cls, K: x, V: x
    x_cls = self.act(self.fc1_cls(x_cls))
    x_cls = self.act_dropout(x_cls)
    x_cls = self.fc2_cls(x_cls)

    x = x + x_cls.clone()
    x = self.act(self.fc1(x))
    x = self.act_dropout(x)
    x = self.fc2(x)
  
    x = x + residual
    return x

    

class Gen(nn.Module):
  def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, dropout_gen, marginal_prob_std, **kwargs):
    '''
    Transformer encoder model
    Arguments:
    n_dim = number of features
    l_dim_gen = dimensionality to embed input
    hidden_gen = dimensionality of hidden layer
    num_layers_gen = number of encoder blocks
    heads_gen = number of parallel attention heads to use
    dropout_gen = regularising layer
    marginal_prob_std = standard deviation of Gaussian perturbation captured by SDE
    '''
    super().__init__()
    self.embed   = nn.Linear(n_dim, l_dim_gen)
    self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=64), nn.Linear(64, 64))
    self.dense1  = Dense(64, 1)
    self.encoder = nn.ModuleList(
        [
           Transformer(
               embed_dim = l_dim_gen,
               num_heads = heads_gen,
               hidden    = hidden_gen,
               dropout   = dropout_gen,
            )
            for i in range(num_layers_gen)
        ]
    )
    self.dropout = nn.Dropout(dropout_gen)
    self.out = nn.Linear(l_dim_gen, n_dim)

    self.cls_token = nn.Parameter(torch.ones(1,1,l_dim_gen), requires_grad=True)
    self.act = nn.GELU()
    self.act_sig = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t, e, mask = None):
    # x: shape(batch_size, n_hits, n_features)
    # t: shape(batch_size)
    x = self.embed(x)

    embed_t_ = self.act_sig(self. embed_t(t)) #shape(batch_size, 64)
    x += self.dense1(embed_t_).clone()

    embed_e_ = self.act_sig(self.embed_t(e))
    x += self.dense1(embed_e_).clone()

    x_cls = self.cls_token.expand(x.size(0), 1, -1) #shape(batch_size, 1, l_dim_gen)
    
    for layer in self.encoder:
      x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)
    
    mean_, std_ = self.marginal_prob_std(x,t)
    return self.out(mean_) / std_[:, None, None]
  
class CNNnet(nn.Module):

    def __init__(self):
        super(CNNnet,self).__init__()

        self.conv1=nn.Conv2d(in_channels=4,out_channels=3,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=3,out_channels=9,kernel_size=5)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(9*4*4,100)
        self.fc2=nn.Linear(100,50)
        self.fc3=nn.Linear(50,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

def loss_fn(model, x, incident_energies, marginal_prob_std, eps=1e-5, device='cpu'):
  '''
  The loss function for training score-based generative models
  Uses the weighted sum of denoising score matching objectives
  Denoising score matching
  - Perturbs data points with pre-defined noise distribution
  - Uses score matching objective to estimate the score of the pertubed data distribution
  - Pertubation avoids need to calculate trace of Jacobian of model output

  Args:
      model: A PyTorch model instance that represents a time-dependent score-based model
      x: A mini-batch of trainging data
      marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
      eps: A tolerance value for numerical stability
  '''
  
  padding_mask = (x[:,:,0]==-20).type(torch.bool)
  # Inverse mask to ignore for when 0-padded hits should be ignored
  output_mask = (x[:,:,0]!=-20).type(torch.int)
  output_mask = output_mask.unsqueeze(-1)
  output_mask = output_mask.expand(output_mask.size()[0], output_mask.size()[1],4)
  model.to(device)
  random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps # shape: (batch_size)
  #random_t = torch.rand(150, device=device) * (1. - eps) + eps
  #print("I am new!")
  incident_energies = torch.squeeze(incident_energies, -1) # shape: (batch_size)
  incident_energies.to(device)
  z = torch.randn_like(x, device=device)*output_mask # shape: (batch_size, n_hit, n_feature)
  mean_, std_ = (x, random_t) # mean_ : x, std_ : shape(batch_size)
  #print(random_t.shape,z.shape,std_.shape)
  std_.to(device)
  perturbed_x = x + z * std_[:, None, None]
  #model_output = model(4,4)
  model_output = model(perturbed_x, random_t, incident_energies, mask=padding_mask)
  losses       = (model_output*std_[:,None,None] + z)**2
  losses = losses*output_mask
  batch_loss   = torch.mean(losses, dim=(0,1,2)) # shape(batch_size)
  return batch_loss

def pc_sampler(score_model, marginal_prob_std, diffusion_coeff, sampled_energies, sampled_hits, batch_size=1, snr=0.16, device='cuda',eps=1e-3):
  '''
  Generate samples from score based models with Predictor-Corrector method
  Args:
  score_model: A pyTorch model that represents the time-dependent score-based model
  marginal_prob_std: A function that gives the std of the perturbation kernel
  diffusion_coeff: A function that gives the diffusion coefficient
  batch_size: The number of samplers to generate by calling this function once.
  num_steps: The number of sampling steps.
             Equivalent to the number of discretized time steps
  device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
  eps: The smallest time step for numerical stability

  Returns:
    samples
  '''
  num_steps = 100
  t = torch.ones(batch_size, device=device)
  gen_n_hits = int(sampled_hits.numpy().max())
  init_x = torch.randn(batch_size, gen_n_hits, 4, device=device)
  mean_, std_ = marginal_prob_std(init_x, t)
  std_.to(device)
  init_x = init_x*std_[:,None,None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x

  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=init_x.device) * time_step
      
      sampled_energies = sampled_energies.to(x.device, torch.float32)
      alpha = torch.ones_like(torch.tensor(time_step))

      grad = score_model(x, batch_time_step, sampled_energies)
      noise = torch.randn_like(x)

      flattened_scores = grad.reshape(grad.shape[0], -1)
      grad_norm = torch.linalg.norm(flattened_scores, dim=-1).mean()
      flattened_noise = noise.reshape(noise.shape[0], -1)
      noise_norm = torch.linalg.norm(flattened_noise, dim=-1).mean()
      langevin_step_size = (snr * noise_norm / grad_norm)**2 * 2 * alpha
   
      x_mean = x + langevin_step_size* grad
      x = x_mean + torch.sqrt(2 * langevin_step_size) * noise

      drift, diff = diffusion_coeff(x, batch_time_step)
      x_mean = x + (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies) * step_size
      x = x_mean + torch.sqrt(diff**2 * step_size)[:, None, None] * torch.randn_like(x)

  return x_mean

def main():
  usage=''
  argparser = argparse.ArgumentParser(usage)
  argparser.add_argument('-o','--output',dest='output_path', help='Path to output directory', default='./', type=str)
  argparser.add_argument('-s','--switches',dest='switches', help='Binary representation of switches that evaluation plots, training, sampling, evaluation plots', default='0000', type=str)
  args = argparser.parse_args()
  workingdir = args.output_path
  switches_ = int('0b'+args.switches,2)
  switches_str = bin(int('0b'+args.switches,2))
  trigger = 0b0001
  print(f'switches_str: {switches_str}')
  print(f'trigger: {trigger}')
  if switches_ & trigger:
    print('input_feature_plots = ON')
  if switches_>>1 & trigger:
    print('training_switch = ON')
  if switches_>>2 & trigger:
    print('sampling_switch = ON')
  if switches_>>3 & trigger:
    print('evaluation_plots_switch = ON')

  print('torch version: ', torch.__version__)
  global device
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Running on device: ', device)
  if torch.cuda.is_available():
    print('Cuda used to build pyTorch: ',torch.version.cuda)
    print('Current device: ', torch.cuda.current_device())
    print('Cuda arch list: ', torch.cuda.get_arch_list())

  print('Working directory: ', workingdir)

  load_n_clouds = 1
  sigma = 25.0
  vesde = utils.VESDE(device=device)
  batch_size = 150
  
  new_marginal_prob_std_fn = functools.partial(vesde.marginal_prob)
  new_diffusion_coeff_fn   = functools.partial(vesde.sde)

  training_file_path = '/afs/cern.ch/user/t/tihsu/ML_hackathon/toy_model/dataset/toy_model.pt'
  filename = training_file_path
  loaded_file = torch.load(filename, map_location=torch.device(device))
  training_data = loaded_file[:10000]
  testing_data  = loaded_file[10000:]
  dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)

  if switches_ & trigger:

    output_directory = workingdir + '/result/'

    all_x = []
    all_y = []
    all_z = []
    all_e = []
    entries = []

    for i, data in enumerate(dataloader,0):
      valid_event = []
      data_np = data.cpu().numpy().copy()
      mask = data_np[:,:,0] != 0
      for j in range(len(data_np)):
        valid_event += ((data_np[j][mask[j]]).tolist())
      valid_event = np.array(valid_event)

      all_e += ((valid_event).copy()[:,0]).flatten().tolist()
      all_x += ((valid_event).copy()[:,1]).flatten().tolist()
      all_y += ((valid_event).copy()[:,2]).flatten().tolist()
      all_z += ((valid_event).copy()[:,3]).flatten().tolist()
      entries += np.count_nonzero(data_np.copy()[:,:,0], axis = -1).tolist()

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit entries')
    plt.hist(entries, 200, range=(0,65), label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_entries.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit energy / (incident energy * 1000)')
    plt.hist(all_e, 200, range=(0,10), label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_energies.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit x position')
    plt.hist(all_x, 50, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_x.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit y position')
    plt.hist(all_y, 50, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_y.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit z position')
    plt.hist(all_z, 89, range=(0, 44), label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_z.png')

  if switches_>>1 & trigger:
    output_directory = os.path.join(workingdir, 'training_result')
    os.system(f'mkdir -p {output_directory}')

    lr = 0.0001
    n_epochs = 50

    model = Gen(4, 200, 128, 3, 1, 0, marginal_prob_std=new_marginal_prob_std_fn)

    print('model: ', model)

    optimiser = Adam(model.parameters(), lr=lr)

    av_losses_per_epoch = []

    for epoch in range(0, n_epochs):
      print(f"epoch: {epoch}")
      batch_losses = []
      batch_counter = 0
      cloud_counter = 0
      cumulative_epoch_loss = 0
      
      process = psutil.Process(os.getpid())
      print('Memory usage of current python process: ', process.memory_info().rss)

      for i, data in enumerate(dataloader,0):
        incident_energies = torch.ones(len(data),device=device)
        batch_loss = loss_fn(model, data, incident_energies, new_marginal_prob_std_fn, device=device)
        batch_counter += 1
        cloud_counter += len(data)
        print(f'Batch: {batch_counter}: loss average: ', batch_loss.item()/len(data))
        optimiser.zero_grad()
        batch_loss.backward(retain_graph=True)
        cumulative_epoch_loss += batch_loss.item()
        optimiser.step()

      av_losses_per_epoch.append(cumulative_epoch_loss/cloud_counter)
      print(f'End-of-epoch: average loss = {av_losses_per_epoch}')
      # Save checkpoint file after each epoch
      torch.save(model.state_dict(), os.path.join(output_directory,'ckpt_tmp_'+str(epoch)+'.pth'))


    print('plotting: ', av_losses_per_epoch)
    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.plot(av_losses_per_epoch, label='training')
    plt.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_directory, 'loss_v_epoch.png'))

  if switches_>>2 & trigger:
    output_directory = os.path.join(workingdir, 'sampling')
    if not os.path.exists(output_directory):
      os.makedirs(output_directory)

    load_name = '/eos/user/t/tihsu/SWAN_projects/ML_hackathon/toy_model/training_result/ckpt_tmp_199.pth'
    sample_batch_size = 500
    model = Gen(4, 200, 128, 3, 1, 0, marginal_prob_std=new_marginal_prob_std_fn)
    model.load_state_dict(torch.load(load_name, map_location=device))
    model.to(device)
    in_energies = []

    hits_lengths = []
    sampled_energies = torch.ones(sample_batch_size)
    sampled_hits = torch.ones(sample_batch_size)*60
    samples = pc_sampler(model, new_marginal_prob_std_fn, new_diffusion_coeff_fn, sampled_energies, sampled_hits, sample_batch_size, device=device)
    print(samples)

if __name__=='__main__':
  main()
