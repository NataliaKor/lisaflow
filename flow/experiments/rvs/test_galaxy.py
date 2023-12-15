"""
Test how sampling fr
"""
import argparse
import numpy as np

from galaxy import Galaxy
#from test import Test

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density_galaxy.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()
    gal = Galaxy(args.config)
    gal.load_fit('minmax_galaxy_sangria.txt')
    samples = gal.sample(5000000)
    print('samples = ', samples)

    log_prob = gal.log_prob(samples)
    print('log_prob = ', log_prob)
    
    #samples_np = samples.get()
    # Save samples to npy file
    #np.save('samples_galaxy_93_1e5.npy', samples_np)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'sample galaxy')
    main(parser)



