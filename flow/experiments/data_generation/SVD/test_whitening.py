'''

  Check if SVD decomposition is good enough.
  Use iterative PCA to add more elements.

'''
import torch
import torch.nn as nn
import sys
import argparse
#sys.path.append('../mbhbs/')
from data_generation.mbhbs.mbhb_model_Lframe import MBHB_gpu, sample_noise
#from mbhb_model_Lframe import MBHB_gpu

from flow.utils.torchutils import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import offsetbox
import h5py
import cupy as xp



def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../../configs/mbhbs/mbhb_resample_all.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='../../configs/mbhbs/mbhb_data_radler_no_time_dist.yaml',
                        help='Path to config file specifying parameters of the source when we sample on the fly')
  
    args = parser.parse_args()

    # Load config
    config = get_config(args.config)
    config_data = get_config(args.config_data)


    # Choose CPU or GPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        dtype = torch.cuda.FloatTensor
    else:
        dev = "cpu"
        dtype = torch.FloatTensor
    print('device = ', dev)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
   
    # Initialise the class for MBHB waveforms
    mbhb = MBHB_gpu(config, config_data, dtype)
    mbhb.freqwave_AET(1)

    # Noise PSD
    dt  = config_data['tvec']['dt']
    freqs, psd_A, psd_E = mbhb.get_noise_psd()    
    df = freqs[2] - freqs[1] 

    # 1. Generate frequency waveform
    A, E = mbhb.get_AE_freq()

    # 2a. Add noise sampled from PSD
    A_noisy = A + sample_noise(psd_A, df)
    E_noisy = E + sample_noise(psd_E, df)

    # 3a. Whiten the waveform with the noise
    A_noisy_white = A_noisy*xp.sqrt(2.0*df)/xp.sqrt(psd_A) #xp.sqrt(2.0*dt)/xp.sqrt(psd_A)
    E_noisy_white = E_noisy*xp.sqrt(2.0*dt)/xp.sqrt(psd_E)

    # 2b. Whiten the waveform
    A_white = A*xp.sqrt(2.0*df)/xp.sqrt(psd_A) #xp.sqrt(2.0*dt)/xp.sqrt(psd_A)
    E_white = E*xp.sqrt(2.0*dt)/xp.sqrt(psd_E)

    # 3b. Add normal noise
    A_white_noisy = xp.random.normal(loc=0.0, scale=1.0, size=freqs.shape) + A_white
    E_white_noisy = xp.random.normal(loc=0.0, scale=1.0/np.sqrt(2.0*df), size=freqs.shape) + E_white

    # Plot amplitudes of the two waveforms to check if they are consistent with each other
    plt.figure()
    plt.loglog(freqs.get(), np.abs(A_noisy_white[0,:].get()), label='first noise then white')
    plt.loglog(freqs.get(), np.abs(A_white_noisy[0,:].get()), label='first white then noise') 
    plt.legend()
    plt.savefig('Test_white.png') 


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)


















