import h5py, math, torch, fnmatch, os
import numpy as np
from torch.nn.functional import normalize
from torch.utils.data import Dataset

class cloud_dataset(Dataset):
  def __init__(self, filename, transform=None, transform_y=None, device='cpu'):
    loaded_file = torch.load(filename, map_location=torch.device(device))
    self.data = loaded_file[0]
    print(f'Loading {filename}: {type(loaded_file[0])}, {type(loaded_file[1])}')
    
    if 'toy_model' in filename:
      self.condition = loaded_file[1].clone().detach()
      self.min_y = torch.min(self.condition)
      self.max_y = torch.max(self.condition)
    else:
      self.condition = torch.as_tensor(loaded_file[1]).float()
      self.min_y = torch.min(self.condition)
      self.max_y = torch.max(self.condition)

    self.transform = transform
    self.transform_y = transform_y
    self.device = device

  def __getitem__(self, index):
    x = self.data[index]
    y = self.condition[index]
    if self.transform:
        x = self.transform(x,y,self.device)
    if self.transform_y:
       y = self.transform_y(y, self.min_y, self.max_y)
    return x,y
  
  def __len__(self):
    return len(self.data)

class rescale_conditional:
  '''Convert hit energies to range |01)
  '''
  def __init__(self):
            pass
  def __call__(self, conditional, emin, emax):
     e0 = conditional
     u0 = (e0-emin)/(emax-emin)
     return u0

class rescale_energies:
        '''Convert hit energies to range |01)
        '''
        def __init__(self):
            pass

        def __call__(self, features, condition, device='cpu'):
            Eprime = features[:,0]/(2*condition)
            alpha = 1e-06
            x = alpha+(1-(2*alpha))*Eprime
            rescaled_e = torch.log(x/(1-x))
            rescaled_e = torch.nan_to_num(rescaled_e)
            rescaled_e = torch.reshape(rescaled_e,(-1,))
            #print(f'features[:,1] {type(features[:,1])}, {features[:,1].shape}: {features[:,1]}')
            x_ = normalize(features[:,1], dim=0)
            y_ = normalize(features[:,2], dim=0)
            z_ = features[:,3]
            # Stack tensors along the 'hits' dimension -1 
            stack_ = torch.stack((rescaled_e,x_,y_,z_), -1)
            self.features = stack_
            
            return self.features

class unscale_energies:
        '''Undo conversion of hit energies to range |01)
        '''
        def __init__(self):
            pass

        def __call__(self, features, condition):
            alpha = 1e-06
            eR = torch.exp(features[:,0])
            A = eR/(1+eR)
            rescaled_e = (A-alpha)*(2*condition)/(1-(2*alpha))

            x_ = features[:,1]
            y_ = features[:,2]
            z_ = features[:,3]
            
            # Stack tensors along the 'hits' dimension -1 
            stack_ = torch.stack((rescaled_e,x_,y_,z_), -1)
            self.features = stack_
            
            return self.features

class VESDE:
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, device='cuda'):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.N = N

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x, device=x.device)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
  
class VPSDE:
  def __init__(self, beta_min=0.01, beta_max=20, N=1000, device='cuda'):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: smallest beta.
      beta_max: largest beta.
      N: number of discretization steps
    """
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps
