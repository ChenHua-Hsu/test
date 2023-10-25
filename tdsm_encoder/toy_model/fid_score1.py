import time, functools, torch, os, random, fnmatch, psutil, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adamax
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#from trans_tdsm import Gen, loss_fn, pc_sampler
#import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import ignite
import torchvision.transforms as transforms
import torchvision.utils as vutils

from collections import OrderedDict
from ignite.metrics import FID, InceptionScore
import torch
from torch import nn, optim

from ignite.engine import Engine

import ignite.metrics.gan.fid as fid


def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

# create default model for doctests

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(1, 1)),
    ('fc', nn.Linear(1, 1))
]))

file_path='model_state_dict.pt'
if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
    loaded_model_state_dict = torch.load('model_state_dict.pt')
    default_model.load_state_dict(loaded_model_state_dict)
else:
    print(f"The file '{file_path}' does not exist.")
    torch.save(default_model.state_dict(), 'model_state_dict.pt') 



#manual_seed(666)

#torch.save(default_model.state_dict(), 'model_state_dict.pt')

# Now, in subsequent runs, load the model's state dictionary
#loaded_model_state_dict = torch.load('model_state_dict.pt')
class Score ():
    def __init__(self,original_energy, original_x, original_y, original_z, all_e_g, all_x_g, all_y_g, all_z_g) :
        self.original_energy=original_energy
        self.original_x=original_x
        self.original_y=original_y
        self.original_z=original_z
        self.all_e_g=all_e_g
        self.all_x_g=all_x_g
        self.all_y_g=all_y_g
        self.all_z_g=all_z_g

    def FID_score(self):
        original_energy = torch.Tensor(self.original_energy).reshape(-1, 1)
        all_e_g = torch.Tensor(self.all_e_g).reshape(-1, 1)
        original_x = torch.Tensor(self.original_x).reshape(-1, 1)
        all_x_g = torch.Tensor(self.all_x_g).reshape(-1, 1)
        original_y = torch.Tensor(self.original_y).reshape(-1, 1)
        all_y_g = torch.Tensor(self.all_y_g).reshape(-1, 1)
        original_z = torch.Tensor(self.original_z).reshape(-1, 1)
        all_z_g = torch.Tensor(self.all_z_g).reshape(-1, 1)

    # Create separate FID metric instances and feature extractor models for each data set
        default_model = nn.Sequential(OrderedDict([
        ('base', nn.Linear(1, 1)),
        ('fc', nn.Linear(1, 1))
        ]))
        #torch.save(default_model.state_dict(), 'model_state_dict.pt')
        file_path='model_state_dict.pt'
        if os.path.exists(file_path):
            print(f"The file '{file_path}' exists.")
            loaded_model_state_dict = torch.load('model_state_dict.pt')
            default_model.load_state_dict(loaded_model_state_dict)
        else:
            print(f"The file '{file_path}' does not exist.")
            torch.save(default_model.state_dict(), 'model_state_dict.pt') 
        #loaded_model_state_dict = torch.load('model_state_dict.pt')
        #default_model.load_state_dict(loaded_model_state_dict)

        metric_e = FID(num_features=1,feature_extractor=default_model )
        default_model.load_state_dict(loaded_model_state_dict)
        metric_x = FID(num_features=1, feature_extractor=default_model)
        default_model.load_state_dict(loaded_model_state_dict)
        metric_y = FID(num_features=1, feature_extractor=default_model)
        default_model.load_state_dict(loaded_model_state_dict)
        metric_z = FID(num_features=1, feature_extractor=default_model)

    # Attach the metrics to separate evaluators
        evaluator_e = Engine(eval_step)
        metric_e.attach(evaluator_e, "fid")

        evaluator_x = Engine(eval_step)
        metric_x.attach(evaluator_x, "fid")

        evaluator_y = Engine(eval_step)
        metric_y.attach(evaluator_y, "fid")

        evaluator_z = Engine(eval_step)
        metric_z.attach(evaluator_z, "fid")

    # Run the evaluators for each data set
        if (original_energy.size()[0]<=all_e_g.size()[0]):
            state_e = evaluator_e.run([[original_energy, all_e_g[:original_energy.size()[0]]]])
            state_x = evaluator_x.run([[original_x, all_x_g[:original_x.size()[0]]]])
            state_y = evaluator_y.run([[original_y, all_y_g[:original_y.size()[0]]]])
            state_z = evaluator_z.run([[original_z, all_z_g[:original_z.size()[0]]]])

        if (original_energy.size()[0]>all_e_g.size()[0]):
            state_e = evaluator_e.run([[original_energy[:all_e_g.size()[0]], all_e_g]])
            state_x = evaluator_x.run([[original_x[:all_x_g.size()[0]], all_x_g]])
            state_y = evaluator_y.run([[original_y[:all_y_g.size()[0]], all_y_g]])
            state_z = evaluator_z.run([[original_z[:all_z_g.size()[0]], all_z_g]])

    # Print the FID scores for each data set
        print("FID of e : ", state_e.metrics["fid"])
        print("FID of x : ", state_x.metrics["fid"])
        print("FID of y : ", state_y.metrics["fid"])
        print("FID of z : ", state_z.metrics["fid"])

        with open("E_Min_XY_Power_3000_fid_values.txt", "w") as file:
        # Write the FID values to the file
            file.write("FID of e : " + str(state_e.metrics["fid"]) + "\n")
            file.write("FID of x : " + str(state_x.metrics["fid"]) + "\n")
            file.write("FID of y : " + str(state_y.metrics["fid"]) + "\n")
            file.write("FID of z : " + str(state_z.metrics["fid"]) + "\n")

        return (state_e.metrics["fid"], state_x.metrics["fid"], state_y.metrics["fid"], state_z.metrics["fid"])