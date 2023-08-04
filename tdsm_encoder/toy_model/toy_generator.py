import torch
import os

def generate_event(max_energy, nevent, nlayer=3):
  Event = []
  max_hits = int((1+nlayer)*nlayer*10/2)
  hits_energy = torch.rand(20000)*max_energy
  hits_energy_np = hits_energy.numpy()
  for nEvent in range(20000):
    hit = []
    remaining_hits = max_hits
    for layer in range(nlayer):
      z = layer + 1
      nhits = torch.randint(low=1, high=z*10, size=(1,))
      energy_dep  = torch.rand((nhits[0],1))*(hits_energy_np[nEvent]/float(max_hits))
      position_xy = torch.randn((nhits[0],2))*z
      position_z  = torch.ones((nhits[0],1))*z
      position    = torch.cat([energy_dep,position_xy, position_z], axis=-1)
      hit.append(position)
      remaining_hits -= nhits
    padding = torch.ones((remaining_hits,4))*-20
    hit.append(padding)
    Event.append(torch.cat(hit, axis=0))
  if not os.path.exists('dataset'):
    os.system("mkdir -p dataset")
  torch.save([Event,hits_energy], 'dataset/toy_model.pt')
  return [Event, hits_energy]

if __name__ == "__main__":
  generate_event(max_energy = 200, nevent = 20000)
