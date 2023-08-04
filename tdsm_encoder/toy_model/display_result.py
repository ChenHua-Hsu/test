import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch 
import pandas as pd
import plotly.graph_objs as go
import random
from scipy import integrate 
from scipy.fft import fft, ifft


def display_result(nevent):
  tensor_list = torch.load("dataset/toy_model.pt")[0]
  np_list = []
  for tensor in tensor_list[0:nevent]:
    np_list.append((tensor.numpy()))
  points = np.concatenate(np_list)
  Energy = points[:,0]
  x      = points[:,1]
  y      = points[:,2]
  z      = points[:,3]
  w      = Energy

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  img = ax.scatter(x, y, z, c=w/5)
  fig.colorbar(img)
  ax.view_init(15,45)
  

def summarize_result(samples, energy, threshold=0.1, label = ""):
    
    global all_x 
    all_x = []
    global all_y 
    all_y = []
    global all_z 
    all_z = []
    global all_e 
    all_e = []
    entries = []
    inject_energy = []
    lin=np.linspace(0,20,10000,endpoint=True)
    print("this is new")
    
    

    for i, data in enumerate(samples,0):
        
        valid_event = []
        data_np   = data.cpu().numpy().copy()
        energy_np = energy.cpu().numpy().copy()
        mask = data_np[:,3] > threshold
        valid_event = data_np[mask]

        if len(valid_event) == 0:
            continue
        valid_event = np.array(valid_event)
        
        all_e += ((valid_event).copy()[:,0]).flatten().tolist()
        all_x += ((valid_event).copy()[:,1]).flatten().tolist()
        all_y += ((valid_event).copy()[:,2]).flatten().tolist()
        all_z += ((valid_event).copy()[:,3]).flatten().tolist()
        entries.append(len(valid_event))
        inject_energy.append(energy_np[i])
    #all_x = [1/20*float(x)+1 for x in all_x]
    #all_y = [1/20*float(x)+1 for x in all_y]   
    
    
    
    fig, ax = plt.subplots(2,3, figsize=(12,12))

    ax[0][0].set_ylabel('# entries')
    ax[0][0].set_xlabel('Hit entries')
    ax[0][0].hist(entries, 200, range=(0,200), label=label)
    ax[0][0].legend(loc='upper right')

    ax[0][1].set_ylabel('# entries')
    ax[0][1].set_xlabel('Hit energy / (incident energy * 1000)')
    ax[0][1].hist(all_e, 50, range=(-0.5,0.5), label=label)
    ax[0][1].legend(loc='upper right')

    ax[0][2].set_ylabel('# entries')
    ax[0][2].set_xlabel('Injection energy')
    ax[0][2].hist(inject_energy, 50, range=(0,200), label=label)
    ax[0][2].legend(loc='upper right')

    ax[1][2].set_ylabel('# entries')
    ax[1][2].set_xlabel('Hit x position')
    ax[1][2].hist(all_x, 50, range=(-0.5,0.5), label=label)
    ax[1][2].legend(loc='upper right')

    ax[1][0].set_ylabel('# entries')
    ax[1][0].set_xlabel('Hit y position')
    ax[1][0].hist(all_y, 50, range=(-0.5,0.5), label=label)
    ax[1][0].legend(loc='upper right')

    ax[1][1].set_ylabel('# entries')
    ax[1][1].set_xlabel('Hit z position')
    ax[1][1].hist(all_z, 50, range=(0.0,2.0), label=label)
    ax[1][1].legend(loc='upper right')
    beta=all_x
    fig.show()
    fig.savefig("test.png")


if __name__ == "__main__":
  display_result(10)
  plt.savefig("demo.png")
 
