from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import random
import time
import json
import re
import math

from sklearn.cluster import KMeans

import torch as th
from torch.utils.data import DataLoader

from utils import tiled_net_out

from data import VolumeDataset

from func_eval import trilinear_f_interpolation,finite_difference_trilinear_grad

from siren import FieldNet, compute_num_neurons
from net_coder import SirenDecoder

#==============================================================================

# Collect arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--volume', required=True, help='path to volumetric dataset')
parser.add_argument('--compressed', required=True, help='path to compressed file')
parser.add_argument('--recon', default='recon', help='path to reconstructed file output')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
parser.set_defaults(cuda=False)

# Create an object to store the arguments
opt = parser.parse_args()
print(opt)

# Instantiates a SirenDecoder object
decoder = SirenDecoder()

# Load a trained version of the network by decoding the file 'opt.compressed' 
net = decoder.decode(opt.compressed)

# Configure network according to the cuda flag
if opt.cuda: net = net.cuda()

# Set the model into evaluation mode; equivalent to self.train(False)
net.eval()

# Load the input volume using numpy, convert it from a numpy array to a PyTorch
# Tensor and then compute the number of scalar entries in the input volume
np_volume = np.load(opt.volume).astype(np.float32)
volume = th.from_numpy(np_volume)
vol_res = th.prod(th.tensor([val for val in volume.shape])).item()

# Compute the size of the volume in bytes (float32 uses 4 bytes per value)
v_size = vol_res*4

# Compute the size (in bytes) of 'opt.compressed' according to the OS
compressed_size = os.path.getsize(opt.compressed)

# Compute the actual compression ratio and print to the console
cr = v_size/compressed_size
print('compression ratio:',cr)

# Compute the maximum and minimum entries and normalise the input volume
raw_min = th.tensor([th.min(volume)],dtype=volume.dtype)
raw_max = th.tensor([th.max(volume)],dtype=volume.dtype)
volume = 2.0*((volume-raw_min)/(raw_max-raw_min)-0.5)

# VolumeDataset is a function from file 'data.py' -> create data object
dataset = VolumeDataset(volume,16)

# Reconstruct the input volume using decoded weights and biases, compute PSNR,
# and saves the reconstructed volume as a .VTK file
tiled_net_out(dataset,
              net, 
              opt.cuda, 
              gt_vol=volume, 
              evaluate=True, 
              write_vols=True, 
              filename=opt.recon)
#==============================================================================