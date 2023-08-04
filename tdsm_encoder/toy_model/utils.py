import h5py, math, torch, fnmatch, os
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset
#from torch_geometric.data import Dataset

class cloud_dataset(Dataset):
  #def __init__(self, data, condition, transform=None):
  def __init__(self, filename, transform=None, device='cpu'):
    loaded_file = torch.load(filename, map_location=torch.device(device))
    self.data = loaded_file[0]
    #self.condition = torch.LongTensor(loaded_file[1]).to(device)
    self.condition = torch.tensor(loaded_file[1], dtype=torch.long, device=torch.device(device))
    self.transform = transform
  def __getitem__(self, index):
    x = self.data[index]
    y = self.condition[index]
    if self.transform:
        x = self.transform(x,y)
    return x,y
  def __len__(self):
    return len(self.data)

class HitCloudDataset(Dataset):
  def __init__(self, datafolder):
    '''Inheriting from pytorch Dataset class
    Loads each file lazily only when __getitem__ is
    called. Returns all graphs in files which can
    then be looped over during training.
    '''
    self.files_list = []
    self.data_folder = datafolder
    for filename in os.listdir(datafolder):
      if fnmatch.fnmatch(filename, 'dataset_2_1_graph*.pt'):
          self.files_list.append(os.path.join(datafolder,filename))
    #with open(self.files_list, 'r') as f:
    #  self.file_list = f.read().splitlines()
  
  def __len__(self):
    return len(self.files_list)
  
  def __getitem__(self,idx):
    filename = self.files_list[idx]
    loaded_file = torch.load(filename)
    return loaded_file

class rescale_energies:
        '''Convert hit energies to range |01)
        '''
        def __init__(self):
            pass

        def __call__(self, features, condition):
            rescaled_e = features.x[:,0]/(condition/1000)
            x_ = features.x[:,1]
            y_ = features.x[:,2]
            z_ = features.x[:,3]
            
            # Stack tensors along the 'hits' dimension -1 
            stack_ = torch.stack((rescaled_e,x_,y_,z_), -1)
            self.features = Data(x=stack_)
            
            return self.features

class uniform_energy_sampler:
    def __init__ (self, filename, sample_batch_size):
        ''' 
        '''
        file_ = h5py.File(filename, 'r')
        self.energies = file_['incident_energies']
        self.min_energy = min(file_['incident_energies'][:])
        self.max_energy = max(file_['incident_energies'][:])
        # Get (1 x sample_batch_size) matrix of numbers in uniform range [0,1)
        sampled_energies_ = torch.rand(sample_batch_size)
        # Produce numbers in the range [0, max_energy-min_energy)
        energy_span = (self.max_energy-self.min_energy)
        # Produce numbers uniformaly distributed between min and max
        self.energy_samples = (energy_span[0] * sampled_energies_) + self.min_energy

    def __len__(self,idx):
        return len(self.energy_samples)

    def __getitem__(self,idx):
        energy_sample_ = self.energy_samples[idx]
        return energy_sample_

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