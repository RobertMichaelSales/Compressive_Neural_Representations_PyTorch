import sys
import argparse
import os
import numpy as np
import time
import json
import re
import math
import struct

from sklearn.cluster import KMeans

import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn

from siren import FieldNet

#==============================================================================
# Define a function that returns a list of tensors of the network weights
# weight_mats[0][1].cpu().shape  = torch.Size([ 68,  3])
# weight_mats[1][1].cpu().shape  = torch.Size([ 68, 68])
# weight_mats[-1][1].cpu().shape = torch.Size([  1, 68])

def get_weight_mats(net):
    
    # Create a list of tuples of the form (layer_name, weight_tensor).
    # weight_mats[0] = ('net_layers.0.linear.weight', [m by n tensor])
    weight_mats = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.weight', name, re.I)]
    
    # Return a list just weights for each consecutive network layer. 
    # The ".cpu()" call is removes tensor "device = cuda:0" for CPU operation    
    return [mat[1].cpu() for mat in weight_mats]

#==============================================================================
# Define a function that returns a list of tensors of the network biases
# bias_vecs[0][1].cpu().shape = torch.Size([68])
# bias_vecs[1][1].cpu().shape = torch.Size([68])
# bias_vecs[-1][1].cpu().shape = torch.Size([1])

def get_bias_vecs(net):
    
    # Create a list of tuples of the form (layer_name, bias_vector).
    # bias_vecs[0] = ('net_layers.0.linear.bias', [vector])
    bias_vecs = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.bias', name, re.I)]
    
    # Return a list just biases for each consecutive network layer. 
    # The ".cpu()" call is removes tensor "device = cuda:0" for CPU operation 
    return [bias[1].cpu() for bias in bias_vecs]

#==============================================================================
# Define a function to perform 1D K-means clustering for quantisation of middle
# network layers. See SirenEncoder.encode() below

# Here, 'w' and 'q' represent 'weight_mats[i]' and 'n_clusters' respectively

def kmeans_quantization(w,q):
    
    # Flatten and convert the weight matrix into a numpy array of float 32
    # i.e. 'w' is of size [68,68] -> 'weight_feat' is of size [68*68,1]
    weight_feat = w.view(-1).unsqueeze(1).numpy()
    
    # The 'n_init' parameter specifies the number of times the KMeans algorithm
    # will be run with different centriod seeds
    
    # The 'n_clusters' parameter specifies the number of clusters to form, thus
    # the number of centroids to generate (default = 512)
    
    # The '.fit()' method computes the k-means clustering
    kmeans = KMeans(n_clusters=q,n_init=4).fit(weight_feat)

    # The '.labels_' method returns a list of the labels: each element recieves 
    # a label to indicate the index of the cluster it has been assigned to
    
    # The '.cluster_centers_' method returns a list of cluster centroids which,
    # in this case, are a 1-dimensional list of coordinates.
    
    return kmeans.labels_.tolist(),kmeans.cluster_centers_.reshape(q).tolist()

#==============================================================================
# Define a function

# Here, 'all_ints' is the list of labels generated by the k-means algorithm and
# the argument 'n_bits' is the number of individual bits used to represent each
# value in the list of labels: # i.e. all_ints[i] = 1 -> '0b000010000' if using
# 'n_bits' = 9 (thus f_str = '#011b')

# The '0b', representing a binary bitstream, is removed when each label integer
# is added to 'bit_string'. The length of the bit_string is divided by 8 to get
# the number of bytes of information.

# The bytearray 'the_bytes' contains a byte-string literal (prefixed with  b'')
# which produces an instance of the 'bytes' type instead of the 'str' type. Any
# byte with a value greater than 128 must be expressed with escapes. By default
# python uses UTF-8 encoding for the bytes in the bytearray.

def ints_to_bits_to_bytes(all_ints,n_bits):
    
    # Define a format string: default for 'n_bits' = 9 is the string '#011b'
    f_str = '#0'+str(n_bits+2)+'b'
    
    # Encodes the list of labels as a bit-string (of length len(labels)*n_bits)
    bit_string = ''.join([format(v, f_str)[2:] for v in all_ints])
    
    # Calculates the number of bytes required to store bit_string
    n_bytes = len(bit_string)//8
    
    # Check if 'n_bytes' exactly divides 'len(bit_string)' -> if not, incriment
    the_leftover = len(bit_string)%8>0
    if the_leftover: n_bytes+=1
        
    # Create an empty bytearray object
    the_bytes = bytearray()
    
    # Iterate across all the bytes in the bit-string
    for b in range(n_bytes):
            
        # Snip out consecutive 8-bit bytes from the bit-string
        bin_val = bit_string[8*b:] if b==(n_bytes-1) else bit_string[8*b:8*b+8]
        
        # Append the base-10 representation of the 8-bit bytes
        the_bytes.append(int(bin_val,2))
    
    return the_bytes,the_leftover

#==============================================================================
# Define a container class (with dictionary functionality) to hold arbitrary...
# attributes that can be dynamically changed during runtime by the user/program

class SimpleMap(dict):
    
    def __init__(self):
        pass

    # The 'getattr' magic method intercepts inexistent attribute lookups. If...
    # the object attribute does exist then '__getattr__' will not be invoked:
    # i.e. 'getattr(obj,"name")' == 'obj["name"]' == 'obj.name'
    def __getattr__(self, attr):
        return self.get(attr)

    # The 'setattr' magic method is always called when setting obj' attributes:
    # i.e. 'setattr(obj,"name",val)' == 'obj["name"] = val' == 'obj.name = val'
    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    # The 'setitem' magic method is always called when assigning values to obj'
    # attributes. The '.update' method updates the dictionary containing the...
    # object's internal variables. This method is only called in '__setattr__'.
    def __setitem__(self, key, value):
        super(SimpleMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

#==============================================================================

class SirenEncoder:
    def __init__(self,net,config):
        self.net = net
        self.config = config
    

    def encode(self,filename,n_bits,d_in=3):
        
        # Default for 'n_bits' is 9 -> default for 'n_clusters' is 512
        n_clusters = int(math.pow(2,n_bits))

        # Make local copies of the relevant network config
        n_layers = self.config['n_layers']
        layers = self.config['layers']
        is_residual = 1 if self.config['is_residual'] else 0
        d_out = 1

        # Get a list of weight matrices and of bias vectors
        weight_mats = get_weight_mats(self.net)
        bias_vecs = get_bias_vecs(self.net)

        # Open a file in 'write in binary' mode 
        file = open(filename,'wb')
        
        #======================================================================
        
        # The module 'struct' converts between Python values and C structures
        # represented as Python bytes opjects. These byte objects are written
        # to a file for later access.
        # -> header: number of layers (integer -> unsigned char)
        header = file.write(struct.pack('B', n_layers))
        # -> header: d_in (integer -> unsigned char)
        header += file.write(struct.pack('B', d_in))
        # -> header: d_out (integer -> unsigned char)
        header += file.write(struct.pack('B', d_out))
        # -> header: is_residual (integer -> unsigned char)
        header += file.write(struct.pack('B', is_residual))
        # -> header: layers (integer -> unsigned int)
        header += file.write(struct.pack(''.join(['I' for _ in range(len(layers))]), *layers))
        # -> header: number of bits for clustering (integer -> unsigned char)
        header += file.write(struct.pack('B', n_bits))

        #======================================================================

        # First layer: weight matrix and bias vector
        # -> Convert weight matrix and bias vectors to lists
        w_pos,b_pos = weight_mats[0].view(-1).tolist(),bias_vecs[0].view(-1).tolist()
        # -> Specify float format for len() floats: 'ff...f'
        w_pos_format = ''.join(['f' for _ in range(len(w_pos))])
        b_pos_format = ''.join(['f' for _ in range(len(b_pos))])
        # -> Write to file for later access (first layer)
        first_layer = file.write(struct.pack(w_pos_format, *w_pos))
        first_layer += file.write(struct.pack(b_pos_format, *b_pos))

        #======================================================================

        # Middle layers: cluster, store clusters, map matrix indices to indices
        mid_bias,mid_weight=0,0
        # Iterate through a zipped list of layer weights and biases
        for weight_mat,bias_vec in zip(weight_mats[1:-1],bias_vecs[1:-1]):
            
            # Compute the labels and centroid coords of the k-means clustering 
            labels,centers = kmeans_quantization(weight_mat,n_clusters)

            # Reassign the centroid coords
            w = centers
            # -> Specify float format for len() floats: 'ff...f'
            w_format = ''.join(['f' for _ in range(len(w))])
            # -> Write to file for later access (middle layers)
            mid_weight += file.write(struct.pack(w_format, *w))
            
            # Returns a bytearray of all labels and an 'is_leftover' flag
            weight_bin,is_leftover = ints_to_bits_to_bytes(labels,n_bits)
            
            # -> Write to file for later access (middle layers)
            mid_weight += file.write(weight_bin)

            # Check if 'n_bits' is wholly divisible by 8
            if n_bits%8 != 0:
                
                # Encode any non-power-2 labels as 16-bit integers at the end
                mid_weight += file.write(struct.pack('I', labels[-1]))
            
            # Convert the 'bias_vec' tensor to a Python list (of size 68)
            b = bias_vec.view(-1).tolist()
            # -> Specify float format for len() floats: 'ff...f'
            b_format = ''.join(['f' for _ in range(len(b))])
            # -> Write to file for later access (middle layers)
            mid_bias += file.write(struct.pack(b_format, *b))
            
        #======================================================================

        # Last layer: weight matrix and bias vector 
        # -> Convert weight matrix and bias vectors to lists
        w_last,b_last = weight_mats[-1].view(-1).tolist(),bias_vecs[-1].view(-1).tolist()
        # -> Specify float format for len() floats: 'ff...f'
        w_last_format = ''.join(['f' for _ in range(len(w_last))])
        b_last_format = ''.join(['f' for _ in range(len(b_last))])
        # -> Write to file for later access (first layer)
        last_layer = file.write(struct.pack(w_last_format, *w_last))
        last_layer += file.write(struct.pack(b_last_format, *b_last))
        
        #======================================================================

        file.flush()
        file.close()

#==============================================================================

class SirenDecoder:
    def __init__(self):
        pass
    #

    def decode(self,filename):
        #weight_mats = get_weight_mats(self.net)
        #bias_vecs = get_bias_vecs(self.net)

        file = open(filename,'rb')

        # header: number of layers
        self.n_layers = struct.unpack('B', file.read(1))[0]
        # header: d_in
        self.d_in = struct.unpack('B', file.read(1))[0]
        # header: d_out
        self.d_out = struct.unpack('B', file.read(1))[0]
        # header: is_residual
        self.is_residual = struct.unpack('B', file.read(1))[0]
        # header: layers
        self.layers = struct.unpack(''.join(['I' for _ in range(self.n_layers)]), file.read(4*(self.n_layers)))
        # header: number of bits for clustering
        self.n_bits = struct.unpack('B', file.read(1))[0]
        self.n_clusters = int(math.pow(2,self.n_bits))
        print('n bits?',self.n_bits,'n clusters?',self.n_clusters)

        # create net from header
        opt = SimpleMap()
        self.d_in = 3
        opt.d_in = self.d_in
        opt.d_out = self.d_out
        opt.L = 0
        opt.w0 = 30
        opt.n_layers = self.n_layers
        opt.layers = self.layers
        opt.is_residual = self.is_residual==1

        net = FieldNet(opt)

        # first layer: matrix and bias
        w_pos_format = ''.join(['f' for _ in range(self.d_in*self.layers[0])])
        b_pos_format = ''.join(['f' for _ in range(self.layers[0])])
        w_pos = th.FloatTensor(struct.unpack(w_pos_format, file.read(4*self.d_in*self.layers[0])))
        b_pos = th.FloatTensor(struct.unpack(b_pos_format, file.read(4*self.layers[0])))

        all_ws = [w_pos]
        all_bs = [b_pos]

        # middle layers: cluster, store clusters, then map matrix indices to indices
        total_n_layers = 2*(self.n_layers-1) if self.is_residual==1 else self.n_layers-1
        for ldx in range(total_n_layers):
            # weights
            n_weights = self.layers[0]*self.layers[0]
            weight_size = (n_weights*self.n_bits)//8
            if (n_weights*self.n_bits)%8 != 0:
                weight_size+=1
            c_format = ''.join(['f' for _ in range(self.n_clusters)])
            centers = th.FloatTensor(struct.unpack(c_format, file.read(4*self.n_clusters)))
            inds = file.read(weight_size)
            bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
            w_inds = th.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])

            if self.n_bits%8 != 0:
                next_bytes = file.read(4)
                w_inds[-1] = struct.unpack('I', next_bytes)[0]
            #

            # bias
            b_format = ''.join(['f' for _ in range(self.layers[0])])
            bias = th.FloatTensor(struct.unpack(b_format, file.read(4*self.layers[0])))

            w_quant = centers[w_inds]
            all_ws.append(w_quant)
            all_bs.append(bias)
        #

        # last layer: matrix and bias
        w_last_format = ''.join(['f' for _ in range(self.d_out*self.layers[-1])])
        b_last_format = ''.join(['f' for _ in range(self.d_out)])
        w_last = th.FloatTensor(struct.unpack(w_last_format, file.read(4*self.d_out*self.layers[-1])))
        b_last = th.FloatTensor(struct.unpack(b_last_format, file.read(4*self.layers[-1])))

        all_ws.append(w_last)
        all_bs.append(b_last)

        wdx,bdx=0,0
        for name, parameters in net.named_parameters():
            if re.match(r'.*.weight', name, re.I):
                w_shape = parameters.data.shape
                parameters.data = all_ws[wdx].view(w_shape)
                wdx+=1
            #
            if re.match(r'.*.bias', name, re.I):
                b_shape = parameters.data.shape
                parameters.data = all_bs[bdx].view(b_shape)
                bdx+=1
            #
        #

        return net
    #
#
