from __future__ import print_function
import argparse
import torch as th
import torch.nn as nn
import numpy as np

#==============================================================================
# Defines a class for 'Sine Layer' objects

class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        
        # Use 'super()' to call the __init__() function of the super-class
        super().__init__()
        
        # Assign values to internal variables
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        # Define a linear layer that maps 'in_features' -> 'out_features'
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Call the 'init_weights' function to initialise layer weights
        self.init_weights()

    #==========================================================================
    # Defines a function to randomly initialise the layer weights

    def init_weights(self):
        
        # Disable gradient calculation to be able to initialise layer weights
        with th.no_grad():
            
            # Set the weights randomly between specified limits
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    
    #==========================================================================
    # Defines a function that performs forward propagation of an input vector
    
    def forward(self, input):
        return th.sin(self.omega_0 * self.linear(input))
    
#==============================================================================
# Defines a class for combining 'Sine Layers' using residual skip connections

class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
    
    
    #==========================================================================
    # Defines a function to randomly initialise the layer weights

    def init_weights(self):
        
        # Disable gradient calculation to be able to initialise layer weights
        with th.no_grad():
        
            # Set the weights randomly between specified limits
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)

    #==========================================================================
    # Defines a function that performs forward propagation of an input vector

    def forward(self, input):
        
        sine_1 = th.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = th.sin(self.omega_0 * self.linear_2(sine_1))
        
        return self.weight_2*(input+sine_2)
    
#==============================================================================
# Defines a function that computes the fewest neurons per layer, for a specific 
# network setup, that achieves the desired target compression ratio.

def compute_num_neurons(opt,target_size):
    
    # Make note of the input and output dimensions
    d_in = opt.d_in                                         # 3 (3D coordinate)
    d_out = opt.d_out                                       # 1 (Scalar)
    
    #==========================================================================
    # Defines a function to calculate the number of total neurons in a network, 
    # given the number of layers, the neurons per layer, and other net options
    
    def network_size(neurons):
        
        # Construct a list containing all the layer dimensions / neurons
        layers = [d_in]
        layers.extend([neurons]*opt.n_layers)
        layers.append(d_out)
        
        # Make note of the total number of trainable layers
        n_layers = len(layers)-1

        # Set the number of total parameters to zero
        n_params = 0
        
        # Iterate through the list containing all the layer dimensions
        for ndx in np.arange(n_layers):
            
            # Retrieve the current layer's input and output dimensions            
            layer_in = layers[ndx]
            layer_out = layers[ndx+1]

            # Compute the dimension of the intermediate residual layer
            og_layer_in = max(layer_in,layer_out)

            # If the first or last layers ->
            if ndx==0 or ndx==(n_layers-1): 
                
                # Add the neurons (weights + biases)
                n_params += ((layer_in+1)*layer_out)
            
            # For all but the first and last layers (i.e. residuals) ->
            else:
                
                # Check if the network is using residual layers ->
                if opt.is_residual:
                    
                    # Check if the input and output dimensions are the same ->
                    is_shortcut = layer_in != layer_out
                    
                    if is_shortcut:
                        
                        # Add the weights and biases of the first a
                        n_params += (layer_in * layer_out) + layer_out
                        
                    n_params += (layer_in * og_layer_in) + og_layer_in
                    n_params += (og_layer_in * layer_out) + layer_out
                    
                else:
                    n_params += ((layer_in+1)*layer_out)

        return n_params
    
    #==========================================================================

    min_neurons = 16
    while network_size(min_neurons) < target_size:
        min_neurons+=1
    min_neurons-=1

    return min_neurons
#

class FieldNet(nn.Module):
    def __init__(self, opt):
        super(FieldNet, self).__init__()

        self.d_in = opt.d_in
        self.layers = [self.d_in]
        self.layers.extend(opt.layers)
        self.d_out = opt.d_out
        self.layers.append(self.d_out)
        self.n_layers = len(self.layers)-1
        self.w0 = opt.w0
        self.is_residual = opt.is_residual

        self.net_layers = nn.ModuleList()
        for ndx in np.arange(self.n_layers):
            layer_in = self.layers[ndx]
            layer_out = self.layers[ndx+1]
            if ndx != self.n_layers-1:
                if not self.is_residual:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=ndx==0))
                    continue
                #

                if ndx==0:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=ndx==0))
                else:
                    self.net_layers.append(ResidualSineLayer(layer_in,bias=True,ave_first=ndx>1,ave_second=ndx==(self.n_layers-2)))
                #
            else:
                final_linear = nn.Linear(layer_in,layer_out)
                with th.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / (layer_in)) / 30.0, np.sqrt(6 / (layer_in)) / 30.0)
                self.net_layers.append(final_linear)
            #
        #
    #

    def forward(self,input):
        batch_size = input.shape[0]
        out = input
        for ndx,net_layer in enumerate(self.net_layers):
            out = net_layer(out)
        #
        return out
    #
#
