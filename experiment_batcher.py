""" Created: 31.05.2023  \\  Updated: 31.05.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, sys, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = sorted(glob.glob("/Data/Compression_Datasets/jhtdb_isotropic1024coarse_pressure/snips/jhtdb_isotropic1024coarse_pressure_snip8_config.json"))
    
    # Set experiment number
    experiment_num = 1
    
    # Set counter and total
    count = 1
    total = 1
    
    # Iterate through all inputs
    for input_dataset_config_path in input_dataset_config_paths:
    
        for compression_ratio in np.array([1e+2],dtype="float32"):
            
            for learning_rate in np.array([1e-3],dtype="float32"):
                
                for batch_size in np.array([512],dtype="int32"):     
                                            
                    for hidden_layers in np.array([14],dtype="int32"):
         
                        # Define the dataset config
                        with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
                        
                        # Run the compression experiment
                        runstring = "python NeurComp_SourceCode/train.py " 
                        
                        runstring = runstring + "--volume {:s} ".format(dataset_config["i_filepath"].replace("snip8","snip8_scalars_only"))
                        
                        runstring = runstring + "--d_in {:d} ".format(len(dataset_config["columns"][0]))
                        runstring = runstring + "--d_out {:d} ".format(len(dataset_config["columns"][1]))
                            
                        runstring = runstring + "--grad_lambda {:f} ".format(0)
                            
                        runstring = runstring + "--n_layers {:d} ".format(hidden_layers)
                        runstring = runstring + "--w0 {:f} ".format(1)
                        
                        runstring = runstring + "--compression_ratio {:f} ".format(compression_ratio)
                        
                        runstring = runstring + "--batchSize {:d} ".format(batch_size)
                        runstring = runstring + "--oversample {:d} ".format(1)
                        runstring = runstring + "--num_workers {:d} ".format(8)
                        
                        runstring = runstring + "--lr {:f} ".format(learning_rate)
                        runstring = runstring + "--n_passes {:f} ".format(30)
                        runstring = runstring + "--pass_decay {:f} ".format(2)
                        runstring = runstring + "--lr_decay {:f} ".format(0.5)
                        runstring = runstring + "--gid {:d} ".format(0)

                        runstring = runstring + "--cuda "

                        runstring = runstring + "--is-residual "

                        runstring = runstring + "--enable-vol-debug"        
                        
                        os.system(runstring)
                        count = count + 1 
                    ##
                ##
            ##
        ##
    ##
##
        
#==============================================================================
   
