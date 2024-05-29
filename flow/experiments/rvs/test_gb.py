"""
Test sampling from single GB
"""
import argparse
from flow.utils.torchutils import *

from galactic_binary import GalacticBinary
import corner
from matplotlib import pyplot as plt

def std_get_wrapper(arg):
    return arg

def cuda_get_wrapper(arg):
    return arg.get()

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density/density_chain0c.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()

    gb = GalacticBinary(args.config)
    
    # Choose if we train the network or load trained weights
    train == False
    if train: 
        # Train the network 
        gb.fit()
    else:
        # Load trained network
        gb.load_fit()

    # Set ranges of the distribution
    config = get_config(args.config)

    # Choose CPU or GPU
    if config['gpu'] is not None:
        assert isinstance(config['gpu'], int)
        import cupy as xp
        get_wrapper = cuda_get_wrapper
    else:
        import numpy as xp
        get_wrapper = std_get_wrapper

    # Load parameters to renormalise samples back to physical range
    path_minmax = config['saving']['save_root'] + 'minmax_' + config['saving']['label'] + '.txt'
    param_min, param_max = np.loadtxt(path_minmax)
    gb.set_min(param_min)
    gb.set_max(param_max)

    # Define how many samples you want to produce
    num_samples = 10000
    samples = gb.sample(num_samples)

    # Estimale log probabilities for the samples
    log_prob = gb.log_prob(samples)
 
    # Plot samples to verify
    figure = corner.corner(get_wrapper(samples),
             plot_datapoints=False,
             fill_contours=True,
             bins=50,
             quantiles=[0.68, 0.954, 0.997],
             color='blue',
             plot_density=True)
    plt.savefig('samples.png')
    plt.close()

    # Save samples to npy file
    np.save('samples.npy', get_wrapper(samples))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'sample galaxy')
    main(parser)



