"""
Test how sampling fr
"""
import argparse
from flow.utils.torchutils import *

from galactic_binary import GalacticBinary
import corner
from matplotlib import pyplot as plt
import numpy as np

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density_chain0c.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()
    gb = GalacticBinary(args.config)
    gb.load_fit()
    # Set ranges of the distribution
    config = get_config(args.config)
    gb.set_min(np.zeros(config['model']['base']['params']))
    gb.set_max(np.ones(config['model']['base']['params']))

    samples = gb.sample(1000)
    log_prob = gb.log_prob(samples)
 
    samples_np = samples.get()
    # Plot samples to verify
    figure = corner.corner(samples_np,
             plot_datapoints=False,
             fill_contours=True,
             bins=50,
             quantiles=[0.68, 0.954, 0.997],
             color='blue',
             plot_density=True)
    plt.savefig('samples_chain0c.png')
    plt.close()

    # Save samples to npy file
    #np.save('samples_ZTFJ1539_1000_1e5samp.npy', samples_np)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'sample galaxy')
    main(parser)



