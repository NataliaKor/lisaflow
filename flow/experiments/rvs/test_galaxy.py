"""
Test how sampling fr
"""
import argparse

from galaxy import Galaxy

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()
    gal = Galaxy(args.config)
    gal.load_fit()
    samples, log_prob1 = gal.sample(5)
    print('samples = ', samples)
    print('log_prob1 = ', log_prob1)
    log_prob2 = gal.log_prob(samples)
    print('log_prob2 = ', log_prob2)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'sample galaxy')
    main(parser)



