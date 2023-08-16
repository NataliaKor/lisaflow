"""
Fit to the samples of the galactic binary
"""
import numpy as np
import argparse
from scipy import stats
import h5py

import torch

from flow.utils.monitor_progress import *
from flow.utils.torchutils import *
from flow.distributions.normal import *
from flow.networks.mlp import MLP

from flow_architecture_density import *

from torch.utils.data import DataLoader
from data_loader_sky import NPYDataset

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/gbs/density_sky.yaml',
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

    # Size of the physical parameters
    # num_coeff = feature_size
    features_size = config['model']['base']['params']

    # Define base distribution. At the moment there are 2 options: 
    if config['model']['base']['distribution'] == 1:
        distribution = StandardNormal((features_size,)).to(dev)
  
    transform = create_transform(config).to(dev)
    flow = Flow(transform, distribution).to(dev)

    # Define path 
    checkpoint = torch.load(config['saving']['save_root'] + config['training']['checkpoints'])
    flow.load_state_dict(checkpoint['model_state_dict'])
 
    # Load min and max values to normalise back 
    filename = config['samples']['path']
    dataset_gb = NPYDataset(filename)
    param_min = dataset_gb.samples_min
    param_max = dataset_gb.samples_max 

    flow.eval()
    with torch.no_grad():

        num_samples = 100 # 200
        #samples = flow.sample(num_samples).squeeze().cpu().detach().numpy()
        samples_gpu, log_prob = flow.sample_and_log_prob(num_samples)
        samples = samples_gpu.squeeze().cpu().detach().numpy()
   
        print('log_prob = ', log_prob)
#        for i in range(0, 50):
#            samples_temp = flow.sample(num_samples).squeeze().cpu().detach().numpy()
#            samples = np.vstack([samples, samples_temp])

        for j in range(param_min.shape[0]):
            samples[:,j] = param_min[j] + samples[:,j]*(param_max[j] - param_min[j])
        print('samples = ', samples)

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description = 'sample galaxy')
    #args = parser.parse_args()
    main(parser)




