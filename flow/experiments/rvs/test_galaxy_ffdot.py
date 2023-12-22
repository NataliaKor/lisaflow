"""
Test how sampling fr
"""
import argparse
import numpy as np

from galaxy_ffdot import GalaxyFFdot
#from test import Test

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density_f.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()
    gal = GalaxyFFdot(args.config)
    # Path with the min and max for the distribution of the galaxy
    gal.load_fit()

    path_minmax = 'minmax_ffdot_sangria.txt'
    param_min, param_max = np.loadtxt(path_minmax) 
    gal.set_min(param_min)
    gal.set_max(param_max)

    samples = gal.sample(1000)
    print('samples = ', samples)

    log_prob = gal.log_prob(samples)
    print('log_prob = ', log_prob)

    #samples_np = samples.get()
    # Save samples to npy file
    #np.save('samples_galaxy_93_1e5.npy', samples_np)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Sample galaxy f and fdot')
    main(parser)



