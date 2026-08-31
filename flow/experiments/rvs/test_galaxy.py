"""
Test how sampling fr
"""
import argparse
import numpy as np

from galaxy import Galaxy
from flow.utils.torchutils import *

import cantuccio as ct
import corner
import pandas as pd
from chainconsumer import Chain, ChainConsumer
import matplotlib.pyplot as plt


def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/gbs/density_galaxy.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()
    gal = Galaxy(args.config)

    fit = False
    if fit == True:
      gal.fit('Galaxy')
    else:
      gal.load_fit()

    config = get_config(args.config)   
    param_min, param_max = np.loadtxt(fname = config['saving']['save_root'] + 'minmax_' + config['saving']['label'] + '.txt', delimiter=' ', usecols=(0,1), unpack=True)
    gal.set_min(param_min)
    gal.set_max(param_max)

    samples = gal.sample(5000000)
    print('samples = ', samples)
    log_prob = gal.log_prob(samples)
    print('log_prob = ', log_prob)
    
    samples_np = samples.get()

    npy_path =  '/sps/lisaf/natalia/github/lisaflow/flow/experiments/galaxy/catalogue_dwds_with_weak_interaction_5parameters.npy'
    samples_orig = np.load(npy_path)
    samples_orig[:, 0] = np.log(samples_orig[:, 0]) # Ampliture
    samples_orig[:, 1] = np.log(samples_orig[:, 1]) # Frequency
    samples_orig[:, 2] =  -np.sign(samples_orig[:, 2])*np.log(np.abs(samples_orig[:, 2])) # Frequency derivative
    samples_orig[:, -1] = np.sin(samples_orig[:, -1]) # EquatorialLatitude

    
    # Save samples to npy file
    #np.save('samples_galaxy_93_1e5.npy', samples_np)
    labels = ['lnA','lnf', '-sgn(fdot)ln|fdot|', 'lam', 'sin(beta)']
    data_dict1 = {label: samples_np[:, i] for i, label in enumerate(labels)}

    #fig, axs = ct.cornerplot(data_dict1)
    #fig = corner.corner(samples_np)
    #corner.corner(samples_orig, figure = fig)
    #fig.savefig('samples_fit.png')

    #fig = corner.corner(samples_orig)
    #corner.corner(samples_orig, figure = fig)
    #fig.savefig('samples_orig.png')


    # Convert to DataFrames
    df1 = pd.DataFrame(samples_orig, columns=labels)
    df2 = pd.DataFrame(samples_np, columns=labels)

    # Create chains
    chain1 = Chain(
        samples=df1,
        name="Original",
        color="blue",
    )

    chain2 = Chain(
        samples=df2,
        name="Fit",
        color="red",
    )

    # Add to ChainConsumer
    c = ChainConsumer()
    c.add_chain(chain1)
    c.add_chain(chain2)

    # Plot corner plot
    fig = c.plotter.plot()

    # Optional
    fig.savefig("comparison.png", dpi=300, bbox_inches="tight")

    fig = plt.figure(figsize=(8, 6))
    plt.hist(
    log_prob.get(),
    bins=100,
    density=True,
    histtype="step",
    linewidth=2,
    color="blue")

    plt.savefig('log_prob.png', dpi=300, bbox_inches="tight")



if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Sample galaxy: amplitude and sky localisation')
    main(parser)
