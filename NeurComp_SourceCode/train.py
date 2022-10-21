#==============================================================================
# Original authors: Yuzhe Lu, Kairong Jiang, Joshua A. Levine, Matthew Berger.
# Modifications by: Robert Sales (20.10.2022)
#==============================================================================

from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import random
import time
import json
import re

import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from siren import FieldNet, compute_num_neurons

from utils import tiled_net_out

from data import VolumeDataset

from func_eval import trilinear_f_interpolation,finite_difference_trilinear_grad

#==============================================================================
# Define the function to be run when 'train.py' is called via terminal commands

if __name__=='__main__':
    
    # Collect arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', required=True, help='path to volumetric dataset')

    parser.add_argument('--d_in', type=int, default=3, help='spatial dimension')
    parser.add_argument('--d_out', type=int, default=1, help='scalar field')

    parser.add_argument('--grad_lambda', type=float, default=0, help='lambda term for gradient regularization - if 0, no regularization is performed, default=0')

    parser.add_argument('--n_layers', type=int, default=8, help='number of layers')
    parser.add_argument('--w0', default=30, help='scale for SIREN')

    parser.add_argument('--compression_ratio', type=float, default=50, help='compression ratio')

    parser.add_argument('--batchSize', type=int, default=1024, help='batch size')
    parser.add_argument('--oversample', type=int, default=16, help='how much to sample within batch items')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=5e-5')
    parser.add_argument('--n_passes', type=float, default=75, help='number of passes to make over the volume, default=50')
    parser.add_argument('--pass_decay', type=float, default=20, help='frequency at which to decay learning rate, default=15')
    parser.add_argument('--lr_decay', type=float, default=.2, help='learning rate decay, default=.2')
    parser.add_argument('--gid', type=int, default=0, help='gpu device id')

    parser.add_argument('--network', default='thenet.pth', help='filename to write the network to, default=thenet.pth')
    parser.add_argument('--config', default='thenet.json', help='configuration file containing network parameters, other stuff, default=thenet.json')

    # Set booleans and their defaults
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
    parser.set_defaults(cuda=False)

    parser.add_argument('--is-residual', dest='is_residual', action='store_true', help='use residual connections')
    parser.add_argument('--not-residual', dest='is_residual', action='store_false', help='don\'t use residual connections')
    parser.set_defaults(is_residual=True)

    parser.add_argument('--enable-vol-debug', dest='vol_debug', action='store_true', help='write out ground-truth, and predicted, volume at end of training')
    parser.add_argument('--disable-vol-debug', dest='vol_debug', action='store_false', help='do not write out volumes')
    parser.set_defaults(vol_debug=True)

    opt = parser.parse_args()
    print(opt)
    device = 'cuda' if opt.cuda else 'cpu'
    
    #==========================================================================
    
    # Load volume from path to volumetric data set, convert to Torch tensor
    np_volume = np.load(opt.volume).astype(np.float32)
    volume = th.from_numpy(np_volume)
    print('volume exts',th.min(volume),th.max(volume))
    
    # Compute the number of scalar entries in the input volume
    vol_res = th.prod(th.tensor([val for val in volume.shape])).item()
    
    # Compute the maximum and minimum entries and normalise the input volume
    raw_min = th.tensor([th.min(volume)],dtype=volume.dtype)
    raw_max = th.tensor([th.max(volume)],dtype=volume.dtype)
    volume = 2.0*((volume-raw_min)/(raw_max-raw_min)-0.5)
    
    # Computes the number of neurons from the user-specified compression ratio
    opt.neurons = compute_num_neurons(opt,int(vol_res/opt.compression_ratio))
    
    # Define the overall network structure in terms of neurons per layer
    opt.layers = []
    for idx in range(opt.n_layers): opt.layers.append(opt.neurons)
        
    # Builds the network and tell the model that it is about to be trained
    net = FieldNet(opt)
    if opt.cuda: net.cuda()
    net.train()
    print(net)

    # Sets the optimiser to Adam (with appropriate training parameters)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    # Create a criterion that measures the mean squared error (default)
    criterion = nn.MSELoss()
    if opt.cuda: criterion.cuda()
        
    # Iterate through the network and count the number of trainable parameters
    # numel() returns the number of elements in the input tensor
    num_net_params = 0
    for layer in net.parameters(): num_net_params += layer.numel() 
    print('number of network parameters:',num_net_params,'volume resolution:',volume.shape)
    
    # Calculate the true compression ratio from input volume and parameters
    compression_ratio = vol_res/num_net_params
    print('compression ratio:',compression_ratio)
    
    # Set the seed for generating random numbers (in PyTorch)
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    th.manual_seed(opt.manualSeed)
    
    #==========================================================================
    # Define a function to return a volume, it's number of scalar entries, it's
    # global minimum and maximum, and a data class object called 'dataset'

    def create_data_loading():
        
        new_vol = volume
        v_res = new_vol.shape[0]*new_vol.shape[1]*new_vol.shape[2]
        
        # VolumeDataset is a function from file 'data.py' -> create data object
        dataset = VolumeDataset(new_vol,opt.oversample)
        
        if opt.cuda:
            global_min_bb = dataset.min_bb.cuda()
            global_max_bb = dataset.max_bb.cuda()
            v_res = dataset.vol_res_float.cuda()
            v = new_vol.cuda()
        
        else:
            global_min_bb = dataset.min_bb
            global_max_bb = dataset.max_bb
            v_res = dataset.vol_res_float
            v = new_vol
        
        return v,v_res,global_min_bb,global_max_bb,dataset

    #==========================================================================
    
    # Set counters to zero, start the timer 
    n_seen,n_iter = 0,0
    tick = time.time()
    first_tick = time.time()
    
    # Run the function 'create_data_loading' as defined above
    v,v_res,global_min_bb,global_max_bb,dataset = create_data_loading()
    
    # From PyTorch documentation: The 'DataLoader' function combines a dataset 
    # and a sampler, and provides an iterable over the given dataset. 
    data_loader = DataLoader(dataset, 
                             batch_size=opt.batchSize, 
                             shuffle=True, 
                             num_workers=int(opt.num_workers))
    
    #==========================================================================
    # Enter into a while loop for training

    while True:
        all_losses = []
        epoch_tick = time.time()

        # Variable 'bdx' is batch number and 'data' is a list of two tensors. 
        # The 1st tensor is of size [batch_size,oversample,input_dimension] 
        # The 2nd tensor is of size [batch_size,oversample,input_dimension] 
        # The elements are for 'raw_positions' and 'positions' respectively    
        for bdx, data in enumerate(data_loader):
            
            # Incriment the number of iterations by 1
            n_iter+=1
            
            # Unpack 'raw_positions', i.e. a tensor of indices in [0,149.0] and
            # Unpack 'positions', i.e. the 'raw_positions' normalised to [-1,1]
            # -> (2.0*(raw_positions/149.0)-1.0) = positions            
            raw_positions, positions = data
            
            if opt.cuda:
                raw_positions = raw_positions.cuda()
                positions = positions.cuda()
            
            # Flatten the position tensors into tensors of size [x,3] where 'x'
            # is equal to (batch_size * oversample) The '.view' function is the 
            # same as 'np.reshape()'.
            raw_positions = raw_positions.view(-1,3)
            positions = positions.view(-1,3)
            
            # If gradient regularisation is not zero, or every 100 batches 
            if (opt.grad_lambda>0) or (bdx%100==0): 
                positions.requires_grad = True

            # Trilinear interpolation approximates at an intermediate point ...
            # within the local axial rectangular prism linearly, using function
            # data on the lattice points.
            
            # Note: in practice, since they sample values at grid points, this 
            # is not really performing interpolation; but, the option is there.            
            field = trilinear_f_interpolation(raw_positions,
                                              v,
                                              global_min_bb,
                                              global_max_bb,
                                              v_res)
            
            # (Re)set the gradient of all the network parameters to zero, then
            # make a prediction of the volume for the current batch positions.        
            net.zero_grad()
            predicted_vol = net(positions)
            predicted_vol = predicted_vol.squeeze(-1)

        
            # If gradient regularisation is being used ->     
            if opt.grad_lambda > 0:
                
                # Compute the target gradients at each of the raw positions in
                # the current batch.                
                target_grad = finite_difference_trilinear_grad(raw_positions,
                                                               v,
                                                               global_min_bb,
                                                               global_max_bb,
                                                               v_res,
                                                               scale=dataset.scales)
            
                # Create a tensor of 'ones' the same shape as 'predicted_vol'                
                ones = th.ones_like(predicted_vol)
            
                # The function 'tf.autograd.grad' computes and returns the sum 
                # of gradients of outputs with respect to the inputs.            
                vol_grad = th.autograd.grad(outputs=predicted_vol,
                                            inputs=positions,
                                            grad_outputs=ones,
                                            retain_graph=True, 
                                            create_graph=True, 
                                            allow_unused=False)[0]
                
                # Compute the loss (mean squared error) of the gradient
                grad_loss = criterion(vol_grad,target_grad)
        
            # Calculate the number of times the entire volume has been swept 
            # across
            n_prior_volume_passes = int(n_seen/vol_res)

            # Calculate the loss (mean squared error) of the field
            vol_loss = criterion(predicted_vol,field)
            
            # Calculate the number grid points/elements that have been swept across
            n_seen += field.view(-1).shape[0]

            # Every 100 batches ->
            if bdx%100==0:
                
                # If gradient regularisation is NOT being used ->
                if opt.grad_lambda == 0:
                        
                    # Compute the target gradients at each of the raw positions
                    # in the current batch.
                    target_grad = finite_difference_trilinear_grad(raw_positions,
                                                                   v,
                                                                   global_min_bb,
                                                                   global_max_bb,
                                                                   v_res,
                                                                   scale=dataset.scales)
                
                    # Create a tensor of 'ones' the same shape as 
                    # 'predicted_vol'
                    ones = th.ones_like(predicted_vol)
                    
                    # The function 'tf.autograd.grad' computes and returns the
                    # sum of gradients of outputs with respect to the inputs.
                    vol_grad = th.autograd.grad(outputs=predicted_vol, 
                                                inputs=positions,
                                                grad_outputs=ones, 
                                                retain_graph=True, 
                                                create_graph=True, 
                                                allow_unused=False)[0]
                    
                    # Compute the loss (mean squared error) of the gradient
                    grad_loss = criterion(vol_grad,target_grad)
                    
                                
                # Print current training information            
                tock = time.time()
                print('loss[',(n_seen/vol_res),n_iter,']:',vol_loss.item(),'time:',(tock-tick))
                print('grad loss',grad_loss.item(),'norms',th.norm(target_grad).item(),th.norm(vol_grad).item())
                tick = tock
            
            # Compute the full loss (using volume loss and gradient loss if 
            # using gradient regularisation)                
            full_loss = vol_loss
            
            if opt.grad_lambda > 0: full_loss += opt.grad_lambda*grad_loss
                
            # The '.backward' function computes the gradient of the loss tensor
            # wrt graph 'leaves'. The graph is differentiated using the chain 
            # rule. The function essentially performs backpropagation.            
            full_loss.backward()
        
            # Perform a single optimisation step
            optimizer.step()
        
            # Append the full loss to a list for storage
            all_losses.append(vol_loss.item())
            
            # Calculate the number of times the entire volume has been swept across
            n_current_volume_passes = int(n_seen/vol_res)
            
            # Decay the learning rate after opt.pass_decay number of passes/epochs            
            if n_prior_volume_passes != n_current_volume_passes and (n_current_volume_passes+1)%opt.pass_decay==0:
                
                print('------ learning rate decay ------',n_current_volume_passes)
                for param_group in optimizer.param_groups: param_group['lr'] *= opt.lr_decay

        
            # If the desired number of passes (epochs) have been completed -> 
            # break
    
            if (n_current_volume_passes+1)==opt.n_passes: break

        # If the desired number of passes (epochs) have been completed -> break    
        if (n_current_volume_passes+1)==opt.n_passes: break
        
        # Stop the timer
        epoch_tock = time.time()
    
    # Stop the timer
    last_tock = time.time()

    #==========================================================================
    # If debugging mode -> run function 'tiled_net_out' from 'utils.py'    
    if opt.vol_debug: tiled_net_out(dataset, net, opt.cuda, gt_vol=volume, evaluate=True, write_vols=True)
        
    # Saves the network 'state_dict' dictionary and network options to disk
    th.save(net.state_dict(), opt.network)
    
    #==========================================================================
    # Calculate the runtine
    total_time = last_tock-first_tick
    
    # Save the config as a dictionary in a .json file config file
    config = {}
    config['grad_lambda'] = opt.grad_lambda
    config['n_layers'] = opt.n_layers
    config['layers'] = opt.layers
    config['w0'] = opt.w0
    config['compression_ratio'] = opt.compression_ratio
    config['batchSize'] = opt.batchSize
    config['oversample'] = opt.oversample
    config['lr'] = opt.lr
    config['n_passes'] = opt.n_passes
    config['pass_decay'] = opt.pass_decay
    config['lr_decay'] = opt.lr_decay
    config['is_residual'] = opt.is_residual
    config['is_cuda'] = opt.cuda
    config['time'] = total_time

    json.dump(config, open(opt.config,'w'))
    
#==============================================================================