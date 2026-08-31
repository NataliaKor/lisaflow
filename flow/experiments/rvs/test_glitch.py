"""
Test sampling from glitch distribution
"""
import argparse
#import lisaflow
from flow.utils.torchutils import *

from flow.experiments.density_estimation.glitch.glitch import Glitch
import corner
from matplotlib import pyplot as plt

def std_get_wrapper(arg):
    return arg

def cuda_get_wrapper(arg):
    return arg.get()

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../configs/glitch/density_glitch.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    args = parser.parse_args()

    gl = Glitch(args.config)

    # Choose if we train the network or load trained weights
    train = False
    if train:
        # Train the network 
        gl.fit('Glitch')
    else:
        # Load trained network
        gl.load_fit()

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
    gl.set_min(param_min)
    gl.set_max(param_max)

    # Define how many samples you want to produce
    n_samples = 2000
    samples_log = gl.sample(n_samples)
    ind_beta = np.argwhere(samples_log[:,0].get() >= 0)
    print('ind_beta.shape = ', ind_beta.shape)
    ind_amp = np.argwhere(10**samples_log[:,1].get() <= 0)
    print('ind_amp.shape = ', ind_amp.shape)
    ind =ind_beta# np.union1d(ind_beta, ind_amp)
    print('ind.shape = ', ind.shape)

    samples = 10**samples_log[ind,:]
    n_events = samples.shape[0]
    print('n_events (log(beta)>0) = ', n_events)
    # Sample from the Poisson distribution
    lambda_per_day = 1

    # Convert rate to events per second
    seconds_per_day = 24 * 60 * 60
    lambda_per_second = lambda_per_day / seconds_per_day

    # Simulate inter-arrival times (in seconds)
    # Exponential distribution is used for Poisson process waiting times
    #n_events = 1000
    interarrival_times = np.random.exponential(
        scale=1/lambda_per_second,
        size=n_events
    )

    # Cumulative event times in seconds
    event_times = np.cumsum(interarrival_times)

    print("Event times (seconds):")
    print(event_times)

    last_ind = np.where(event_times > 3600*24*365*2)[0][0]
    print('last_ind = ', last_ind)
    print('last time stemp = ', event_times[last_ind])
    last_ind = last_ind + 10
    # Interferometer
    tm_ind_arr = ['tm_12', 'tm_21', 'tm_13', 'tm_31', 'tm_23', 'tm_32']
    tm_ind = np.random.choice(tm_ind_arr, size=last_ind+1)
    print('tm_ind = ', tm_ind)
    print('type(tm_ind) = ', type(tm_ind))

    # Estimale log probabilities for the samples
    #log_prob = gl.log_prob(samples)
    #print('samples = ', 10**samples)

    print('samples[:last_ind+1,0,0].get() = ', samples[:last_ind+1,0,0].get())

   # with open('glitches.txt', 'w') as f:
   #     for a, b, c, d in zip(samples[:last_ind+1,0,0].get(), samples[:last_ind+1,0,1].get(), event_times[:last_ind+1], tm_ind.tolist()):
   #         f.write(f'{a:e} {b:e} {c:e} {d:s}\n')

    rows = np.array(list(zip(samples[:last_ind+1,0,0].get(), samples[:last_ind+1,0,1].get(), event_times[:last_ind+1], tm_ind.tolist())),dtype=object)
    np.savetxt(
       'glitch.txt',
       rows,
       fmt=['%e', '%e', '%e', '%s'],
       header="Beta[s] Amplitude[m/s2] Time_of_arrival[s] Interferometer")

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
    parser = argparse.ArgumentParser(description = 'sample glitch')
    main(parser)

