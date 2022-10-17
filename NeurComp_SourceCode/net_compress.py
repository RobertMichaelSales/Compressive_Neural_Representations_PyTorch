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

import torch as th
from torch.utils.data import DataLoader

from siren import FieldNet, compute_num_neurons
from net_coder import SirenEncoder

#==============================================================================

# Collect arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--net', required=True, help='path to trained network')
parser.add_argument('--config', required=True, help='path to network config')
parser.add_argument('--compressed', required=True, help='path to compressed file for output')
parser.add_argument('--cluster_bits', type=int, default=9, help='number of bits for cluster (2^b clusters)')

# Create an object to store the arguments
opt = parser.parse_args()
print(opt)

# Load the network config file in read-only mode
config = json.load(open(opt.config,'r'))

# Add other elements to the network config
opt.d_in = 3                                        # input dimensions
opt.d_out = 1                                       # final dimensions
opt.L = 0                                           # (?)
opt.w0 = config['w0']                               # scale for SIREN
opt.n_layers = config['n_layers']                   # number of total layers
opt.layers = config['layers']                       # list of layer dimensions
opt.compression_ratio = config['compression_ratio'] # target compression ratio
opt.oversample = config['oversample']               # oversample setting
opt.cuda = config['is_cuda']                        # cuda flag
opt.is_residual = config['is_residual']             # residual connection flag

# Re-build an empty (untrained) SIREN network from the above-given options
net = FieldNet(opt)

# Load weights and biases from 'opt.net' by means of an ordered dictionary
net.load_state_dict(th.load(opt.net))

# Configure network according to the cuda flag
if opt.cuda: net = net.cuda() 

# Set the model into evaluation mode; equivalent to self.train(False)
net.eval()

# Instantiates a SirenEncoder object
encoder = SirenEncoder(net, config)

# Quantise the weights of the network, saving to opt.compressed (a binary file)
encoder.encode(opt.compressed,opt.cluster_bits)
#==============================================================================