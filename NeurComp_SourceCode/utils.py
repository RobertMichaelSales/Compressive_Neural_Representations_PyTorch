#==============================================================================
# Original authors: Yuzhe Lu, Kairong Jiang, Joshua A. Levine, Matthew Berger.
# Modifications by: Robert Sales (20.10.2022)
#==============================================================================

import numpy as np
import random
import time

import torch as th

from pyevtk.hl import imageToVTK

#==============================================================================
# Define a function that computes/predicts the scalar field and gradient field 
# using the current network weights and biases

def field_and_grad_from_net(dataset, net, is_cuda, tiled_res=32):
    
    # Set 'target_res' to the shape of the input volume, i.e [150,150,150]
    # Target resolution
    target_res = dataset.vol_res
    
    # Set 'full_vol' to be the shape of the input volume filled with zeros
    full_vol = th.zeros(target_res)
    
    # Set 'target_res_pos' to the shape of the input coordinates
    # i.e. [150,150,150,3]
    # Target resolution for positions
    target_res_pos = list(dataset.vol_res)
    target_res_pos.append(net.d_in)
    
    # Set 'full_grad' to be the shape of the input volume filled with zeros
    full_grad = th.zeros(target_res_pos)
    
    # Iterate through the x-slices  (interval = tiled_res)
    # This gives -> i.e. [0,32,64,96,128] not [0,32,64,96,128,150]    
    for xdx in np.arange(0,target_res[0],tiled_res):
        
        x_begin = xdx
        x_end = xdx+tiled_res if xdx+tiled_res <= target_res[0] else target_res[0]
    
        # Iterate through the y-slices  (interval = tiled_res)
        # This gives -> i.e. [0,32,64,96,128] not [0,32,64,96,128,150]
        for ydx in np.arange(0,target_res[1],tiled_res):
            
            y_begin = ydx
            y_end = ydx+tiled_res if ydx+tiled_res <= target_res[1] else target_res[1]
            
            # Iterate through the z-slices  (interval = tiled_res)
            # This gives -> i.e. [0,32,64,96,128] not [0,32,64,96,128,150]
            for zdx in np.arange(0,target_res[2],tiled_res):
                
                z_begin = zdx
                z_end = zdx+tiled_res if zdx+tiled_res <= target_res[2] else target_res[2]
                
                # Calculate the distance between successive points. This equals
                # 'tiled_res' except for the final iteration, which is smaller.
                # i.e. "32, 32, 32, 32, 22"    
                tile_resolution = th.tensor([x_end-x_begin,y_end-y_begin,z_end-z_begin],dtype=th.int)
                
                # Determines normalised coordinates of cubes within the volume 
                # such that the entire volume is higlighted once and only once 
                min_alpha_bb = th.tensor([x_begin/(target_res[0]-1),y_begin/(target_res[1]-1),z_begin/(target_res[2]-1)],dtype=th.float)
                max_alpha_bb = th.tensor([(x_end-1)/(target_res[0]-1),(y_end-1)/(target_res[1]-1),(z_end-1)/(target_res[2]-1)],dtype=th.float)
                
                # Determines the indices (as floats) of the cube corners of the
                # above-mentioned cubes.
                min_bounds = dataset.min_bb + min_alpha_bb*(dataset.max_bb-dataset.min_bb)
                max_bounds = dataset.min_bb + max_alpha_bb*(dataset.max_bb-dataset.min_bb)  
                
                # Disable gradient calculation, which is useful for inference
                # and requires less CPU time per calculation
                
                # Compute normalised coordinates of uniform tile sampling
                # where 'tile_positions' lie in the range [-1,+1]. Returns 
                # a [32,32,32,3] cube of 3D coordinates per iteration
                tile_positions = dataset.scales.view(1,1,1,3)*dataset.tile_sampling(min_bounds,max_bounds,tile_resolution)
                
                if is_cuda: tile_positions = tile_positions.unsqueeze(0).cuda()
                    
                # The 'requires_grad' call states that gradients need computing
                # for the tensor 'tile_positions' 
                tile_positions.requires_grad = True
                
                # Compute/predict the scalar values at the prescribed tile
                # positions
                tile_vol = net(tile_positions.unsqueeze(0)).squeeze(0).squeeze(-1)

                # Create a tensor of 'ones' the same shape as 'tile_vol'
                ones = th.ones_like(tile_vol)
                
                # The function 'tf.autograd.grad' computes and returns the sum 
                # of gradients of outputs with respect to the inputs.
                vol_grad = th.autograd.grad(outputs=tile_vol,
                                            inputs=tile_positions,
                                            grad_outputs=ones,
                                            retain_graph=False,
                                            create_graph=True,
                                            allow_unused=False)[0]
                
                # Fill the relevant tiles within the full gradient with the 
                # above computed scalar values
                full_grad[x_begin:x_end,y_begin:y_end,z_begin:z_end] = vol_grad.detach().cpu()
                
                # Fill the relevant tiles within the full volume with the above 
                # computed scalar values
                full_vol[x_begin:x_end,y_begin:y_end,z_begin:z_end] = tile_vol.detach().cpu()

    return full_vol,full_grad

#==============================================================================
# Define a function that computes/predicts the scalar field (output) using the 
# current network weights and biases

def field_from_net(dataset, net, is_cuda, tiled_res=32, verbose=False):
    
    # Set 'target_res' to the shape of the input volume, i.e [150,150,150]
    # Target resolution
    target_res = dataset.vol_res
    
    # Set 'full_vol' to be the shape of the input volume filled with zeros
    full_vol = th.zeros(target_res)
    
    # Iterate through the x-slices (interval = tiled_res)
    # This gives -> i.e. [0,32,64,96,128] not [0,32,64,96,128,150]
    for xdx in np.arange(0,target_res[0],tiled_res):
        
        if verbose:print('x',xdx,'/',target_res[0])
            
        x_begin = xdx
        x_end = xdx+tiled_res if xdx+tiled_res <= target_res[0] else target_res[0]
        
        # Iterate through the y-slices  (interval = tiled_res)
        # This gives -> i.e. [0,32,64,96,128] not [0,32,64,96,128,150]
        for ydx in np.arange(0,target_res[1],tiled_res):
            
            y_begin = ydx
            y_end = ydx+tiled_res if ydx+tiled_res <= target_res[1] else target_res[1]
            
            # Iterate through the z-slices  (interval = tiled_res)
            # This gives -> i.e. [0,32,64,96,128] not [0,32,64,96,128,150]
            for zdx in np.arange(0,target_res[2],tiled_res):
                
                z_begin = zdx
                z_end = zdx+tiled_res if zdx+tiled_res <= target_res[2] else target_res[2]

                # Calculate the distance between successive points. This equals
                # 'tiled_res' except for the final iteration, which is smaller.
                # i.e. "32, 32, 32, 32, 22"                
                tile_resolution = th.tensor([x_end-x_begin,y_end-y_begin,z_end-z_begin],dtype=th.int)

                # Determines normalised coordinates of cubes within the volume 
                # such that the entire volume is higlighted once and only once 
                min_alpha_bb = th.tensor([x_begin/(target_res[0]-1),y_begin/(target_res[1]-1),z_begin/(target_res[2]-1)],dtype=th.float)
                max_alpha_bb = th.tensor([(x_end-1)/(target_res[0]-1),(y_end-1)/(target_res[1]-1),(z_end-1)/(target_res[2]-1)],dtype=th.float)
                
                # Determines the indices (as floats) of the cube corners of the
                # above-mentioned cubes.
                min_bounds = dataset.min_bb + min_alpha_bb*(dataset.max_bb-dataset.min_bb)
                max_bounds = dataset.min_bb + max_alpha_bb*(dataset.max_bb-dataset.min_bb)  

                # Disable gradient calculation, which is useful for inference
                # and requires less CPU time per calculation
                with th.no_grad():
                    
                    # Compute normalised coordinates of uniform tile sampling
                    # where 'tile_positions' lie in the range [-1,+1]. Returns 
                    # a [32,32,32,3] cube of 3D coordinates per iteration
                    tile_positions = dataset.scales.view(1,1,1,3)*dataset.tile_sampling(min_bounds,max_bounds,tile_resolution)
                
                    if is_cuda: tile_positions = tile_positions.unsqueeze(0).cuda()
                        
                    # Compute/predict the scalar values at the prescribed tile
                    # positions
                    tile_vol = net(tile_positions.unsqueeze(0)).squeeze(0).squeeze(-1)
                    
                    # Fill the relevant tiles within the full volume with the 
                    # above computed scalar values
                    full_vol[x_begin:x_end,y_begin:y_end,z_begin:z_end] = tile_vol.cpu()

    return full_vol

#==============================================================================
# Define a function that reconstructs the input volume using trained weights...
# and biases, computes the PSNR (peak signal-to-noise ratio) and then saves the
#  reconstructed input volume as a .VTK file.

def tiled_net_out(dataset, net, is_cuda, gt_vol=None, evaluate=True, write_vols=False, filename='vol'):
    
    # Sets the network into evaluation mode, equivalent to 'self.train(False)'
    net.eval()
    
    # Compute/predict the entire volume using the current weights and biases
    full_vol = field_from_net(dataset, net, is_cuda, tiled_res=32)
    
    psnr = 0
    print('writing to VTK...')
    
    # Check if in evaluation mode, and check whether 'gt_vol' contains any data
    # The variable 'gt_vol' is the ground-truth volume, i.e. the original input
    if evaluate and gt_vol is not None:
        
        # Calculate the difference between the prediction and ground truth
        diff_vol = gt_vol - full_vol
        
        # Calculate the squared maximum difference
        max_diff = th.max(gt_vol)-th.min(gt_vol)
        
        # Calculate the absolute difference (error)
        l1_diff = th.mean(th.abs(diff_vol))
        
        # Calculate the mean-squared difference (error)
        mse = th.mean(th.pow(diff_vol,2))
        
        # Calculate the peak signal to noise ratio (PSNR)
        psnr = -20.0*th.log10(th.sqrt(mse)/max_diff)
        
        print('PSNR:',psnr,'l1:',l1_diff,'mse:',mse,'rmse:',th.sqrt(mse))

    if write_vols:
        
        # Save the computed/predicted volume to a VTK (or equivalent) file
        imageToVTK(filename, pointData = {'sf':full_vol.numpy()})
        
        # Save the original ground truth volume to VTK if it exists 
        if gt_vol is not None: imageToVTK('gt', pointData = {'sf':gt_vol.numpy()})
        
    print('back to training...')
    
    # Set the model/network/module back into training mode
    net.train()
    
    return psnr
#==============================================================================