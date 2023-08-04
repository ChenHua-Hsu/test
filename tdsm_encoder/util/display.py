import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import pandas as pd
import plotly.graph_objs as go

def plot_distribution(point_clouds_loader, label="", padding_value=-20):    

    all_x = []
    all_y = []
    all_z = []
    all_e = []
    all_incident_e = []
    entries = []
    
    for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0): 
        
        valid_event = []
        data_np = shower_data.cpu().numpy().copy()
        energy_np = incident_energies.cpu().numpy().copy()
        
        mask = ~(data_np[:,:,3] == padding_value)
        
        for j in range(len(data_np)):
            valid_event = data_np[j][mask[j]]
            all_e += ((valid_event).copy()[:,0]).flatten().tolist()
            all_x += ((valid_event).copy()[:,1]).flatten().tolist()
            all_y += ((valid_event).copy()[:,2]).flatten().tolist()
            all_z += ((valid_event).copy()[:,3]).flatten().tolist()
            entries.append(len(valid_event))
            all_incident_e.append(energy_np[j])
           
    
    fig, ax = plt.subplots(2,3, figsize=(12,12))

    ax[0][0].set_ylabel('# entries')
    ax[0][0].set_xlabel('Hit entries')
    ax[0][0].hist(entries, label=label)
    ax[0][0].legend(loc='upper right')

    ax[0][1].set_ylabel('# entries')
    ax[0][1].set_xlabel('Transformed hit energy')
    ax[0][1].hist(all_e, 50, label=label)
    ax[0][1].legend(loc='upper right')

    ax[0][2].set_ylabel('# entries')
    ax[0][2].set_xlabel('Transformed incident energies')
    ax[0][2].hist(all_incident_e, label=label)
    ax[0][2].legend(loc='upper right')

    ax[1][2].set_ylabel('# entries')
    ax[1][2].set_xlabel('Hit x position')
    ax[1][2].hist(all_x, 50, label=label)
    ax[1][2].legend(loc='upper right')

    ax[1][0].set_ylabel('# entries')
    ax[1][0].set_xlabel('Hit y position')
    ax[1][0].hist(all_y, 50, label=label)
    ax[1][0].legend(loc='upper right')

    ax[1][1].set_ylabel('# entries')
    ax[1][1].set_xlabel('Hit z position')
    ax[1][1].hist(all_z, 50, label=label)
    ax[1][1].legend(loc='upper right')
    return fig
    
