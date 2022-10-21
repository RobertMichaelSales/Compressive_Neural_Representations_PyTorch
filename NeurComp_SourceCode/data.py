#==============================================================================
# Original authors: Yuzhe Lu, Kairong Jiang, Joshua A. Levine, Matthew Berger.
# Modifications by: Robert Sales (20.10.2022)
#==============================================================================

import numpy as np
import json
import time
import random

import sys
import torch as th
from torch.utils.data.dataset import Dataset

#==============================================================================
# Define a class to deal with the dataset, i.e. the input volume, variables ...
# there are some random lines of code that don't seem to have any real function

class VolumeDataset(Dataset):
    
    def __init__(self,volume,oversample=16):
        
        # Make note of the volume shape i.e. resolution
        # i.e. [150,150,150]
        self.vol_res = volume.shape
        
        # Make note of the number of elements in the volume
        # i.e. 3375000
        self.n_voxels = th.prod(th.tensor(self.vol_res,dtype=th.int)).item()
        
        # Make note of the volume shape i.e. resolution as a float
        # i.e. [150.,150.,150.]
        self.vol_res_float = th.tensor([self.vol_res[0],self.vol_res[1],self.vol_res[2]],dtype=th.float)
        
        # Make note of the maximum and minimum values
        # i.e. tensor([-1.]) and tensor([+1.])
        self.min_volume = th.tensor([th.min(volume)],dtype=volume.dtype)
        self.max_volume = th.tensor([th.max(volume)],dtype=volume.dtype)

        # Make note of minimum and maximum bounding box values
        # i.e. tensor([0.,0.,0.]) and tensor([149.,149.,149.])
        self.min_bb = th.tensor([0.0,0.0,0.0],dtype=th.float)
        self.max_bb = th.tensor([float(volume.size()[0]-1),float(volume.size()[1]-1),float(volume.size()[2]-1)],dtype=th.float)
        
        # Calculate the diagonal (?) i.e. tensor([149., 149., 149.])
        self.diag = self.max_bb-self.min_bb
        
        self.pos_eps = 1e-8
        self.diag_eps = self.diag*(1.0-2.0*self.pos_eps)

        # Calculate (?) 
        self.max_dim = th.max(self.diag)            # i.e. tensor(149.)
        self.scales = self.diag/self.max_dim        # tensor([1., 1., 1.])
        #self.scales = th.ones(3)

        # Create a tensor of shape [150,150,150,3] with indices of each element
        # i.e. lattice[x][y][z] = tensor([x.,y.,z.]). Note: entries are floats.
        # Entries go from [0.,0.,0.] -> [149.,149.,149.]
        self.lattice = self.tile_sampling(self.min_bb,self.max_bb,res=self.vol_res,normalize=False)
        
        # Create a flattened version of self.lattice. Note: Entries are floats.
        # Entries go from [0.,0.,0.] -> [149.,149.,149.]. Shape is [3375000, 3]
        self.full_tiling = self.tile_sampling(self.min_bb,self.max_bb,res=self.vol_res,normalize=False).view(-1,3)

        # Make note of the number of elements in the field, same as above
        self.actual_voxels = self.full_tiling.shape[0]

        # Make note of the user's requirement for oversampling
        self.oversample = oversample

    #==========================================================================
    # Define a function that generates coordinates for the input volume
    
    def tile_sampling(self, sub_min_bb, sub_max_bb, res=None, normalize=True):
        
        # If 'res' is 'None', then set the res(olution), i.e. to [150,150,150].
        # However, self.tile_res is never actually declared        
        if res is None: 
            res = th.tensor([self.tile_res,self.tile_res,self.tile_res],dtype=th.int)
            
        # Create a tensor of size [150,150,150,3] for the x,y and z coordinates 
        # at each point in the input field
        positional_data = th.zeros(res[0],res[1],res[2],3)

        # 'start' = tensor([0.,0.,0.]) or tensor([0.,0.,0.])       (norm=False)
        # 'end'   = tensor([1.,1.,1.]) or tensor([149.,149.,149.]) (norm=False)      
        start = sub_min_bb / (self.max_bb-self.min_bb) if normalize else sub_min_bb
        end = sub_max_bb / (self.max_bb-self.min_bb) if normalize else sub_max_bb
        
        # Fill with non-normalised or normalised positional (coordinate) data:
        # i.e. positional_data[x][y][z] = tensor([x,y,z])
        # i.e. positional_data[x][y][z] = tensor([x,y,z]) normalised to ->[0,1]
        positional_data[:,:,:,0] = th.linspace(start[0],end[0],res[0],dtype=th.float).view(res[0],1,1)
        positional_data[:,:,:,1] = th.linspace(start[1],end[1],res[1],dtype=th.float).view(1,res[1],1)
        positional_data[:,:,:,2] = th.linspace(start[2],end[2],res[2],dtype=th.float).view(1,1,res[2])

        # Redistribute the normalised data from [0,1] to [-1,1]
        return 2.0*positional_data - 1.0 if normalize else positional_data

    #==========================================================================
    # Define a function that returns a number of random position coordinates
    
    def uniform_sampling(self,n_samples=None):
        
        # If 'n_samples' is 'None' then use the self defined number of samples.
        # self.n_samples is however, not specified anywhere else in the code
        if n_samples is None:
            n_samples = self.n_samples
        
        # Returns 'n_samples' randomly sampled position coordinates from within
        # the input field coordinates.        
        return self.pos_eps + self.min_bb.unsqueeze(0) + th.rand(n_samples,3)*self.diag_eps.unsqueeze(0)
    
    #==========================================================================
    # Customises the built-in len() function
    
    def __len__(self):
        
        # Return the number of (scalars) elements in the input volume
        return self.n_voxels
    
    #==========================================================================
    # Customises the built-in getitem() function
    
    def __getitem__(self, index):
        
        # Randomly choose 'oversample' number of items from 'full_tiling' taken
        # from positions 0 -> 3375000
        # i.e. full_tiling[1000] = tensor([0.,6.,100.])
        random_positions = self.full_tiling[th.randint(self.actual_voxels,(self.oversample,))]
        
        # Normalise 'random_positions'
        normalized_positions = 2.0 * ( (random_positions - self.min_bb.unsqueeze(0)) / (self.max_bb-self.min_bb).unsqueeze(0) ) - 1.0
        
        # Scale along each axis in case the volume is not cubic in shape
        normalized_positions = self.scales.unsqueeze(0)*normalized_positions
        
        return random_positions, normalized_positions

#==============================================================================