import os
import sys
import optparse, argparse
import subprocess
import json
import ROOT
import h5py
import torch
import numpy as np

def GetEntries(pathIn, iin):

  infile = h5py.File(os.path.join(pathIn, iin), 'r')
  return len(infile["showers"])

def MergeGraphList(fOut, fList):
  mergelist_0 = []
  mergelist_1 = []
  for f in fList:
    _l = torch.load(f)
    mergelist_0.extend(_l[0])
    mergelist_1.extend(_l[1])
  mergelist = [mergelist_0, mergelist_1]
  print(np.shape(mergelist))
  torch.save(mergelist, fOut)
    

def prepare_shell(shell_file, command, condor, FarmDir):

  cwd       = os.getcwd()
  with open('%s/%s'%(FarmDir,shell_file), 'w') as shell:
    shell.write('#!/bin/bash\n')
    shell.write('WORKDIR=%s\n'%cwd)
#    shell.write('eval `scram r -sh`\n')
    shell.write('cd ${WORKDIR}\n')
    shell.write(command)
  condor.write('cfgFile=%s\n'%shell_file)
  condor.write('queue 1\n')

if __name__=='__main__':
  
  usage = 'usage: %prog [options]'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-d', '--dataset', dest='dataset', help='dataset[1/2/3]',default=1, type=int)
  parser.add_argument("--coordinate", dest='coordinate', help='[polar/euclidian]', default='polar', type=str)
  parser.add_argument("--zero_pedding", dest='zero_pedding', action="store_true")
  parser.add_argument("--test", action="store_true")
  parser.add_argument('--store_geometric', dest = 'store_geometric', action = "store_true")
  parser.add_argument("--merge", action="store_true")
  args = parser.parse_args()

  dataset_directory = "/eos/user/t/tihsu/SWAN_projects/homepage/datasets/"
  TaskList = { "dataset1": [
                {"particle": "photon", "xml": "binning_dataset_1_photons.xml", "data": ["dataset_1_photons_1.hdf5", "dataset_1_photons_2.hdf5"]}, 
                {"particle": "pion", "xml": "binning_dataset_1_pions.xml", "data": ["dataset_1_pions_1.hdf5"]}],
               "dataset2": [
                {"particle": "electron", "xml": "binning_dataset_2.xml", "data": ["dataset_2_1.hdf5","dataset_2_2.hdf5"]}],
               "dataset3": [
                {"particle": "electron", "xml": "binning_dataset_3.xml", "data": ["dataset_3_1.hdf5","dataset_3_2.hdf5","dataset_3_3.hdf5","dataset_3_4.hdf5"]}]
             }


  FarmDir   = 'Farm'
  cwd       = os.getcwd()
  os.system('mkdir -p %s'%FarmDir)
  if args.dataset == 1:
    batchsize = 5000
  elif args.dataset == 2:
    batchsize = 500
  else:
    batchsize = 100
  if not args.store_geometric:
    if args.dataset == 2:
      batchsize = 10000
    elif args.dataset == 3:
      batchsize = 2000
    else:
      batchsize = 50000

  condor = open('%s/condor.sub'%FarmDir,'w')
  condor.write('output = %s/job_common.out\n'%FarmDir)
  condor.write('error  = %s/job_common.err\n'%FarmDir)
  condor.write('log    = %s/job_common.log\n'%FarmDir)
  condor.write('executable = %s/$(cfgFile)\n'%FarmDir)
  condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
  condor.write('request_GPUs = 1\n')
  condor.write('+JobFlavour = "tomorrow"\n')
#  condor.write('+MaxRuntime = 7200\n')

  if args.zero_pedding:
    pedding_name = 'zero_pedding'
    pedding_command = '--zero_pedding'
  else:
    pedding_name = 'no_pedding'
    pedding_command = ''

  for task in TaskList["dataset%d"%args.dataset]:
    for data in task["data"]:
      nEntries = GetEntries(dataset_directory, data)
      fList = []
      for batch in range(((nEntries//batchsize) + 1)):
        command  = "source /cvmfs/sft.cern.ch/lcg/views/LCG_102b_cuda/x86_64-centos7-gcc8-opt/setup.sh\n"
        if args.store_geometric:
          command += "python GraphCreator.py --particle %s --xml %s --dataset %s --from %d --to %d --tag %d --store_geometric --coordinate %s %s\n"%(task["particle"], task["xml"], os.path.join(dataset_directory, data), batchsize*batch, min(batchsize*(batch+1),nEntries), batch, args.coordinate, pedding_command)
          shell_file = 'graph_%s_%s_%s_%d.sh'%(data, pedding_name, args.coordinate, batch)
          prepare_shell(shell_file, command, condor, FarmDir)
          fList.append(os.path.join(dataset_directory, data.replace('.hdf5','_graph_%s_%s_%d.pt'%(pedding_name, args.coordinate, batch))))
        else:
          command += "python GraphCreator.py --particle %s --xml %s --dataset %s --from %d --to %d --tag %d --coordinate %s %s\n"%(task["particle"], task["xml"], os.path.join(dataset_directory, data), batchsize*batch, min(batchsize*(batch+1),nEntries), batch, args.coordinate, pedding_command)
          shell_file = 'tensor_%s_%s_%s_%d.sh'%(data, pedding_name, args.coordinate, batch)
          prepare_shell(shell_file, command, condor, FarmDir)
        fList.append(os.path.join(dataset_directory, data.replace('.hdf5','_tensor_%s_%s_%d.pt'%(pedding_name, args.coordinate, batch))))
      if args.merge:
        if args.store_geometric:
          from torch_geometric.data import Data
          MergeGraphList(os.path.join(dataset_directory, data.replace('.hdf5','_graph_%s_%s.pt'%(pedding_name, args.coordinate))), fList)
        else:
          MergeGraphList(os.path.join(dataset_directory, data.replace('.hdf5','_tensor_%s_%s.pt'%(pedding_name, args.coordinate))), fList)


  condor.close()
  if not (args.test or args.merge):
    print ("Submitting Jobs on Condor")
    os.system('condor_submit %s/condor.sub'%FarmDir)

    
