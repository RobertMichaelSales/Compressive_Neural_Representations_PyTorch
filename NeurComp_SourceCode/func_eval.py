#==============================================================================
# Original authors: Yuzhe Lu, Kairong Jiang, Joshua A. Levine, Matthew Berger.
# Modifications by: Robert Sales (20.10.2022)
#==============================================================================

import torch as th

#==============================================================================
# Define a function to perform trilinear interpolation on the input field at...
# the raw_positions (indices) provided. The output 'interp_val' is, essentially
# the same as the input 'f' at positions 'p', but with tiny differences ~(1e-8)

# Here:
# -> 'p' is a tensor of raw positions, i.e. indices from 'f' in range [0,149.0]
# -> 'f' is the normalised input volume 'v' or 'volume'
# -> 'min_bb' is the minimum volume indices:    tensor([  0.,  0.,  0.])
# -> 'max_bb' is the maximum volume indices:    tensor([149.,149.,149.])
# -> 'res' is the resolution in each dimension: tensor([150.,150.,150.])

# In this case 'p' is flattened to size (batch_size*oversample,input_dimension)

def trilinear_f_interpolation(p,f,min_bb,max_bb,res):
       
    # Compute normalised lattice positions, i.e. integer coordinates or indices
    normalized_p = ( (p-min_bb.unsqueeze(0))/((max_bb-min_bb).unsqueeze(0)) ) * (res.unsqueeze(0) - 1)
    
    # Compute the floors and ceilings of 'normalized_p', then recast as float64
    lattice_p_floor = th.floor(normalized_p).to(th.long)
    lattice_p_ceil = th.ceil(normalized_p).to(th.long)

    # "it's possible that normalized_p has values that are integers? welp, ok,
    # let's account for that..." - original authors
    
    # Define a small number called 'min_ref = tensor([1.0000e-12])'
    min_ref = 1e-12*th.ones_like(normalized_p[:1,0])
    
    # Compute the elementwise maximum of 'ceil-floor' vs. 'min_ref' 
    the_diff = th.max((lattice_p_ceil-lattice_p_floor).to(th.double),min_ref.to(th.double))

    # Compute a tensor of 0's and 1's, where 1's are found at the differences
    alpha = (normalized_p.to(th.double)-lattice_p_floor.to(th.double)) / the_diff
    
    # Convert 'alpha' from type float64 to float32 and determine the compliment
    alpha = alpha.to(th.float)
    one_alpha = 1.0-alpha

    # X-interpolation (tensor index is 0) -> these are mostly or all? identical
    x_interp_y0z0 = one_alpha[:,0]*f[lattice_p_floor[:,0],lattice_p_floor[:,1],lattice_p_floor[:,2]]+alpha[:,0]*f[lattice_p_ceil[:,0],lattice_p_floor[:,1],lattice_p_floor[:,2]]
    x_interp_y1z0 = one_alpha[:,0]*f[lattice_p_floor[:,0],lattice_p_ceil[:,1],lattice_p_floor[:,2]]+alpha[:,0]*f[lattice_p_ceil[:,0],lattice_p_ceil[:,1],lattice_p_floor[:,2]]
    x_interp_y0z1 = one_alpha[:,0]*f[lattice_p_floor[:,0],lattice_p_floor[:,1],lattice_p_ceil[:,2]]+alpha[:,0]*f[lattice_p_ceil[:,0],lattice_p_floor[:,1],lattice_p_ceil[:,2]]
    x_interp_y1z1 = one_alpha[:,0]*f[lattice_p_floor[:,0],lattice_p_ceil[:,1],lattice_p_ceil[:,2]]+alpha[:,0]*f[lattice_p_ceil[:,0],lattice_p_ceil[:,1],lattice_p_ceil[:,2]]

    # Y-interpolation (tensor index is 1)
    y_interp_z0 = one_alpha[:,1]*x_interp_y0z0+alpha[:,1]*x_interp_y1z0
    y_interp_z1 = one_alpha[:,1]*x_interp_y0z1+alpha[:,1]*x_interp_y1z1

    # Z-interpolation (tensor index is 2) -> final interpolated value
    interp_val = one_alpha[:,2]*y_interp_z0+alpha[:,2]*y_interp_z1

    return interp_val
#==============================================================================
# Define a function that computes gradients using a central differencing scheme

# Here:
# -> 'p' is a tensor of raw positions, i.e. indices from 'f' in range [0,149.0]
# -> 'f' is the normalised input volume 'v' or 'volume'
# -> 'min_bb' is the minimum volume indices:    tensor([  0.,  0.,  0.])
# -> 'max_bb' is the maximum volume indices:    tensor([149.,149.,149.])
# -> 'res' is the resolution in each dimension: tensor([150.,150.,150.])
# -> 'scale' is the scale of the lattice in each dimension: tensor([1.,1.,1.])

# In this case 'p' is flattened to size (batch_size*oversample,input_dimension)

def finite_difference_trilinear_grad(p,f,min_bb,max_bb,res,scale=None):
    
    # This is the most seemingly convoluted way of getting the above tensors
    
    # Compute the x/y/z steps between neighbouring elements in the input volume 
    # -> for the test volume, this gives 'tensor([[1.,1.,1.]])'
    x_step = ((max_bb-min_bb)/(res-1)).unsqueeze(0)
    y_step = ((max_bb-min_bb)/(res-1)).unsqueeze(0)
    z_step = ((max_bb-min_bb)/(res-1)).unsqueeze(0)

    # Zero elements outside of the axis of interest:
    # -> xtensor([[1.,0.,0.]]) / tensor([[0.,1.,0.]]) / tensor([[1.,0.,0.]])
    x_step[:,1:] = 0
    y_step[:, 0] = 0
    y_step[:, 2] = 0
    z_step[:,:2] = 0
    
    # Compute normalised lattice positions, i.e. integer coordinates or indices
    # either side of the coordinates in 'raw_positions'
    x_negative = p-x_step
    x_positive = p+x_step
    y_negative = p-y_step
    y_positive = p+y_step
    z_negative = p-z_step
    z_positive = p+z_step

    # Ensure all of the '_negative' or '_positive' lattice positions are within
    # the minimum and maximum bounds after adding/subtracting '_step' from each
    x_negative[x_negative[:,0] < min_bb[0],0] = min_bb[0]
    y_negative[y_negative[:,1] < min_bb[1],1] = min_bb[1]
    z_negative[z_negative[:,2] < min_bb[2],2] = min_bb[2]
    x_positive[x_positive[:,0] > max_bb[0],0] = max_bb[0]
    y_positive[y_positive[:,1] > max_bb[1],1] = max_bb[1]
    z_positive[z_positive[:,2] > max_bb[2],2] = max_bb[2]

    # Gradients are computed using finite central differencing:
    # -> f'(x) ~ (f(x+deltaX)-f(x-deltaX)) / 2deltaX
    
    # Compute the '2deltaX' denominator of the finite central difference method
    if scale is None:
        x_diff = 2*(x_positive[:,0]-x_negative[:,0]) / (max_bb[0]-min_bb[0])
        y_diff = 2*(y_positive[:,1]-y_negative[:,1]) / (max_bb[1]-min_bb[1])
        z_diff = 2*(z_positive[:,2]-z_negative[:,2]) / (max_bb[2]-min_bb[2])
    else:
        x_diff = 2*scale[0]*(x_positive[:,0]-x_negative[:,0]) / (max_bb[0]-min_bb[0])
        y_diff = 2*scale[1]*(y_positive[:,1]-y_negative[:,1]) / (max_bb[1]-min_bb[1])
        z_diff = 2*scale[2]*(z_positive[:,2]-z_negative[:,2]) / (max_bb[2]-min_bb[2])
    
    # Compute x,y,and z derivatives at each point using trilinear interpolation
    # to first obtain 'f(x+deltaX)' and 'f(x-deltaX)'
    x_deriv = (trilinear_f_interpolation(x_positive,f,min_bb,max_bb,res) - trilinear_f_interpolation(x_negative,f,min_bb,max_bb,res))/x_diff
    y_deriv = (trilinear_f_interpolation(y_positive,f,min_bb,max_bb,res) - trilinear_f_interpolation(y_negative,f,min_bb,max_bb,res))/y_diff
    z_deriv = (trilinear_f_interpolation(z_positive,f,min_bb,max_bb,res) - trilinear_f_interpolation(z_negative,f,min_bb,max_bb,res))/z_diff

    # Return a tensor of size (batch_size*oversample,input_dimension) of x,y...
    # and z derivatives (instead of x,y and z locations)
    return th.cat((x_deriv.unsqueeze(1),y_deriv.unsqueeze(1),z_deriv.unsqueeze(1)),1)

#==============================================================================