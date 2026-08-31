"""
Fit to the samples of the galactic binary
"""
import numpy as np
import argparse
from scipy import stats

import torch

from flow.utils.torchutils import *
from torch.utils.data import DataLoader
from flow.experiments.data_loader_sky import NPYDataset

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--resume', type=int, default=1, help='Flag whether to resume training')

    args = parser.parse_args()

    # Choose CPU or GPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        dtype = torch.cuda.FloatTensor
    else:
        dev = "cpu"
        dtype = torch.FloatTensor
    print('device = ', dev)
    cuda = torch.cuda.is_available()

    # Load config
    config = get_config(args.config)

    # Load min and max values to normalise back 
    filename = config['samples']['path']
    dataset_gb = NPYDataset(filename)
    param_min = dataset_gb.samples_min
    param_max = dataset_gb.samples_max 

    np.savetxt('minmax_galaxy_sangria_log.txt', (param_min, param_max))
 
  
if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description = 'sample galaxy')
    main(parser)




