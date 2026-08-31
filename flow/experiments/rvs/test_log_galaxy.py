"""
Test how sampling fr
"""
import argparse
import numpy as np

from galaxy_log import Galaxylog
#from test import Test

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density_galaxy_log.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()
    gal = Galaxylog(args.config)
    gal.load_fit()
    samples, log_prob1 = gal.sample(100000)
    print('samples = ', samples)
    print('log_prob1 = ', log_prob1)
    log_prob2 = gal.log_prob(samples)
    print('log_prob2 = ', log_prob2)
    samples_np = samples.get()

    # Save samples to npy file
    np.save('samples_galaxy_log_1e5.npy', samples_np)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'sample galaxy')
    main(parser)



