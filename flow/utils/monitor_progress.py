'''
  Code to make corner and pp plots for monitoring the progress.
'''
import numpy as np
import corner
import matplotlib.pyplot as plt
from scipy import stats
from flow.utils.transform_to_as import * 
import h5py


def renormalise_gb_samples(pars, amp_true, freq_min, freq_max):
    '''
      Prior ranges for GB densities. Used to renormalise results of sampling back to physical densities.
      Inputs: 
          pars -- normalised samples
          amp_true -- amplitude value for the injected signal
          freq -- vector of frequencies for injected signal. I might have differet vectors
                  so maybe I should only take the vector in the range.
    '''
    num_params = pars.shape[1]-1
    print('pars.shape = ', pars.shape)
    # Prior ranges for the parameters
    prior = np.zeros((num_params, 2))
    prior[0,:] = np.log10(np.array([amp_true/100, amp_true*100]))
    prior[1,:] = np.array([freq_min, freq_max])
    prior[2,:] = np.array([-1.e-14, 1.e-14])
    prior[3,:] = np.array([-1.0, 1.0])
    prior[4,:] = np.array([0.0, 2.0*np.pi])
    prior[5,:] = np.array([-1.0, 1.0])
    prior[6,:] = np.array([0.0, 2.0*np.pi])
    prior[7,:] = np.array([0.0, 2.0*np.pi])
    
    print('prior[0,:] = ', prior[0,:])

    # Normalise back to the physical parameters range
    pp = np.empty_like(pars[:,:-1])
    for ii in range(num_params):
        pp[:,ii] = prior[ii, 0] + pars[:,ii]*(prior[ii, 1] - prior[ii, 0])
    #pp[:,0] = 10.**pp[:,0]
    #pp[:,3] = np.arcsin(pp[:,3])
    #pp[:,5] = np.arccos(pp[:,5])
    # Return samples in the physical range of parameters
    ind = [1,2,3,4,5,0,7,6]
    return pp[:,ind]


def make_pp(percentiles, parameter_labels, iteration, test_label, ks=True):

    percentiles = percentiles/100.
    nparams = percentiles.shape[-1]
    nposteriors = percentiles.shape[0]

    ordered = np.sort(percentiles, axis=0)
    ordered = np.concatenate((np.zeros((1, nparams)), ordered, np.ones((1, nparams))))
    y = np.linspace(0, 1, nposteriors + 2)

    fig = plt.figure(figsize=(10,10))

    for n in range(nparams):
        if ks:
            pvalue = stats.kstest(percentiles[:, n], 'uniform')[1]
            plt.step(ordered[:, n], y, where='post', label=parameter_labels[n] + r' ({:.3g})'.format(pvalue))
        else:
            plt.step(ordered[:, n], y, where='post', label=parameter_labels[n])
    plt.plot(y, y, 'k--')
    plt.legend()
    plt.ylabel(r'$CDF(p)$')
    plt.xlim((0,1))
    plt.ylim((0,1))

    plt.xlabel(r'$p$')

    ax = fig.gca()
    ax.set_aspect('equal', anchor='SW')

    plt.savefig('../experiments/plots/ppplot_' + str(test_label) + '_' + str(iteration) + '.png')
    plt.close()


def make_cp_compare_samples_gb(flow, iteration, labels, param_mean, param_std, coeff_norm, truths, test_label, filename, amp_true, freq_min, freq_max):

    # TODO
    # This is the type for the files that Stas has provided me.
    # But I need to change it for the type of the files that we get from Eryn.

    samples_comparison =  renormalise_gb_samples(np.load(filename), amp_true, freq_min, freq_max)
    num_samples = 10000

    samples = flow.sample(num_samples, coeff_norm).squeeze()
    samples = samples * param_std + param_mean
    fig = corner.corner(samples_comparison[:,[0,1,2,3]], 
                        labels=labels,
                        show_titles = True,
                        plot_datapoints=False, 
                        fill_contours=True, 
                        bins=50,
                        levels=[0.68, 0.954, 0.997], 
                        color='red',
                        plot_density=True)
 
    corner.corner(samples.cpu().detach().numpy(),
             fig = fig,
             labels=labels,
             show_titles=True, 
             plot_dataponts = False,
             fill_contours=True,
             bins=50,
             levels=[0.68, 0.954, 0.997],
             color='blue',
             plot_density=True,
             truths=truths)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()


def make_cp_compare_samples_mbhb(flow, iteration, labels, param_mean, param_std, coeff_norm, truths, test_label):

    # Samples 
    fl0 = 'samples/MBHB/fullRadNNFL-2D-phi_b_0_resampled_SSBframe_LDC.dat'
    fl1 = 'samples/MBHB/fullRadNNFL-2D-phi_b_1_resampled_SSBframe_LDC.dat'

    dat0 = np.genfromtxt(fl0, delimiter=',', names=True)
    dat1 = np.genfromtxt(fl1, delimiter=',', names=True)
    datBM = np.concatenate((dat0, dat1))

    d_BM = np.zeros((len(datBM), len(lbls)))
    d_BM[:,0] = datBM['Mass1']
    d_BM[:,1] = datBM['Mass2']
    d_BM[:,2] = datBM['Spin1']
    d_BM[:,3] = datBM['Spin2']
    d_BM[:,4] = datBM['CoalescenceTime']
    d_BM[:,5] = datBM['Distance']
    d_BM[:,6] = datBM['Inclination']
    d_BM[:,7] = datBM['EclipticLongitude'] + 2.0*np.pi
    d_BM[:,8] = datBM['EclipticLatitude']

    # Calculate chirp mass and mass ratio
    m1 = d_BM[:,0]
    m2 = d_BM[:,1]
    mu = (m1 * m2)**0.6/(m1 + m2)**0.2
    q = m2/m1 

    psi, inc = ldctools.AziPolAngleL2PsiIncl(beta, lam, the, phi)
    tcL, lamL, betaL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(tc, lam, beta, psi, 0.0)  # Check what exactly is zero value

    # Transform parameters to the LISA frame
    num_samples = 200

    samples = flow.sample(num_samples, coeff_norm).squeeze()

    for i in range(0, 100):
        samples_temp = flow.sample(num_samples, coeff_norm).squeeze()
        samples = torch.vstack([samples, samples_temp])

    for j in range(param_mean.shape[0]):
        samples[:,j] = samples[:,j]*param_std[j] + param_mean[j]

    # Plot a_s at the moment but for the future I need to reconstruct the physical parameters

    figure = corner.corner(samples.cpu().detach().numpy(),
             labels=labels,
             show_titles=True, truths=truths)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()

def make_cp_as_std(flow, iteration, labels, param_mean, param_std, coeff_norm, truths, test_label):
  
    num_samples = 200

    print('coeff_norm.shape = ', coeff_norm.shape)
    samples = flow.sample(num_samples, coeff_norm).squeeze()

    for i in range(0, 100):
        samples_temp = flow.sample(num_samples, coeff_norm).squeeze()
        samples = torch.vstack([samples, samples_temp])

    for j in range(param_mean.shape[0]):
        samples[:,j] = samples[:,j]*param_std[j] + param_mean[j]

    # Plot a_s at the moment but for the future I need to reconstruct the physical parameters
    
    figure = corner.corner(samples.cpu().detach().numpy(),
             labels=labels,
             show_titles=True, truths=truths)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()


def make_cp_as(flow, iteration, labels, param_min, param_max, coeff_norm, truths, test_label, Amax):
  
    num_samples = 200

    samples = flow.sample(num_samples, coeff_norm).squeeze()

    for i in range(0, 100):
        samples_temp = flow.sample(num_samples, coeff_norm).squeeze()
        samples = torch.vstack([samples, samples_temp])

    # Reconstruct back to the physical parameters
    samples[:,0], samples[:,5], samples[:,6], samples[:,7]  = reconstruct_params(samples[:,0], samples[:,5], samples[:,6], samples[:,7], Amax)

    for j in range(param_min.shape[0]):
        samples[:,j] = param_min[j] + samples[:,j]*(param_max[j] - param_min[j])

    figure = corner.corner(samples.cpu().detach().numpy(),
             labels=labels,
             show_titles=True, truths=truths)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()


def make_cp(flow, iteration, labels, param_min, param_max, coeff_norm, truths, test_label):
  
    num_samples = 200

    samples = flow.sample(num_samples, coeff_norm).squeeze().cpu().detach().numpy()

    for i in range(0, 100):
        samples_temp = flow.sample(num_samples, coeff_norm).squeeze().cpu().detach().numpy()
        samples = np.vstack([samples, samples_temp])

    for j in range(param_min.shape[0]):
        samples[:,j] = param_min[j] + samples[:,j]*(param_max[j] - param_min[j])

    figure = corner.corner(samples,
             labels=labels,
             show_titles=True, truths=truths)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()


def make_cp_density_estimation(flow, iteration, labels, param_min, param_max, test_label):
  
    num_samples = 200

    samples = flow.sample(num_samples).squeeze().cpu().detach().numpy()

    for i in range(0, 50):
        samples_temp = flow.sample(num_samples).squeeze().cpu().detach().numpy()
        samples = np.vstack([samples, samples_temp])

    for j in range(param_min.shape[0]):
        #samples[:,j] = param_min[j] + samples[:,j]*(param_max[j] - param_min[j])
        samples[:,j] = param_min[j] + (samples[:,j] + 1.0)*(param_max[j] - param_min[j])/2.0

    #samples_load = np.load('/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/samples/run_1_accepted_rj_points.npy')
    #print('samples_load.shape = ', samples_load.shape)
    #fig = corner.corner(samples_load, 
    #                    labels=labels,
    #                    show_titles = True,
    #                    plot_datapoints=False, 
    #                    fill_contours=True, 
    #                    bins=50,
    #                    quantiles=[0.68, 0.954, 0.997], 
    #                    color='red',
    #                    plot_density=True)
    #plt.savefig('samples_original.png')
    #plt.close()


    #print('samples.shape = ', samples.shape)
    corner.corner(samples, # fig = fig
             labels=labels,
             show_titles=True,
             plot_datapoints=True,
             fill_contours=True, 
             bins=100,
             levels=[0.68, 0.954, 0.997],
             color='blue',
             plot_density=True)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()


def make_cp_density_estimation_minus1(flow, iteration, labels, param_min, param_max, test_label, filename):
# TODO remove loops
# Plot Galaxy for comparison 
  
    num_samples = 200

    samples = flow.sample(num_samples).squeeze().cpu().detach().numpy()

    for i in range(0, 50):
        samples_temp = flow.sample(num_samples).squeeze().cpu().detach().numpy()
        samples = np.vstack([samples, samples_temp])

    for j in range(param_min.shape[0]):
        samples[:,j] = param_min[j] + (samples[:,j] + 1.0)*(param_max[j] - param_min[j])/2.0

    samples_data = np.load(filename)
    #with h5py.File(filename, "r") as f:
    #    load_data = f['gbs_sky_dist'][()]
    #    samples = np.array(load_data.tolist())
    #    samples[:,0] = np.log10(self.samples[:,0])
    #fig = corner.corner(samples_data[:,1:-4], 
    #                       labels = labels,  
    #                       color = 'red',
    #                       plot_datapoints=False,
    #                       fill_contours=True,
    #                       bins = 50,
    #                       levels=[0.68,0.954,0.997],
    #                       weights=np.ones(samples_data.shape[0])/samples_data.shape[0],
    #                       plot_density=True)

    corner.corner(samples, #            fig = fig,
             labels=labels,
             show_titles=True,
             plot_datapoints=False,
             fill_contours=True, 
             bins=50,
             quantiles=[0.68, 0.954, 0.997],
             color='blue',
             weights=np.ones(samples.shape[0])/samples.shape[0],
             plot_density=True)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()


def make_cp_density_estimation_gb_sampling_points(flow, iteration, labels, param_min, param_max, test_label, filename):
# TODO remove loops
# Plot Galaxy for comparison 
  
    num_samples = 200

    samples = flow.sample(num_samples).squeeze().cpu().detach().numpy()

    #for i in range(0, 50):
    #    samples_temp = flow.sample(num_samples).squeeze().cpu().detach().numpy()
    #    samples = np.vstack([samples, samples_temp])
    
    samples = param_min + (samples + 1.0)*(param_max - param_min)/2.0

    samples_data = np.load(filename)
    # Normalise parameters to the physical ranges
    
    fig = corner.corner(samples_data[:,1:-4], 
                           labels = labels,  
                           color = 'red',
                           plot_datapoints=False,
                           fill_contours=True,
                           bins = 50,
                           levels=[0.68,0.954,0.997],
                           weights=np.ones(samples_data.shape[0])/samples_data.shape[0],
                           plot_density=True)

    corner.corner(samples, 
             fig = fig,
             labels=labels,
             show_titles=True,
             plot_datapoints=False,
             fill_contours=True, 
             bins=50,
             quantiles=[0.68, 0.954, 0.997],
             color='blue',
             weights=np.ones(samples.shape[0])/samples.shape[0],
             plot_density=True)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()




def make_cp_density_estimation_minus1_galaxy(flow, iteration, labels, param_min, param_max, test_label, filename):
# TODO remove loops
# Plot Galaxy for comparison 
  
    num_samples = 200

    samples = flow.sample(num_samples).squeeze().cpu().detach().numpy()

    for i in range(0, 50):
        samples_temp = flow.sample(num_samples).squeeze().cpu().detach().numpy()
        samples = np.vstack([samples, samples_temp])

    for j in range(param_min.shape[0]):
        samples[:,j] = param_min[j] + (samples[:,j] + 1.0)*(param_max[j] - param_min[j])/2.0

    samples_data = np.load(filename)
    #with h5py.File(filename, "r") as f:
    #    load_data = f['gbs_sky_dist'][()]
    #    samples = np.array(load_data.tolist())
    #    samples[:,0] = np.log10(self.samples[:,0])
    #samples_data[:,0] = np.log10(samples_data[:,0])
    fig = corner.corner(samples_data, 
                           labels = labels,  
                           color = 'red',
                           plot_datapoints=False,
                           fill_contours=True,
                           bins = 50,
                           levels=[0.68,0.954,0.997],
                           weights=np.ones(samples_data.shape[0])/samples_data.shape[0],
                           plot_density=True)

    corner.corner(samples, 
             fig = fig,
             labels=labels,
             show_titles=True,
             plot_datapoints=False,
             fill_contours=True, 
             bins=50,
             quantiles=[0.68, 0.954, 0.997],
             color='blue',
             weights=np.ones(samples.shape[0])/samples.shape[0],
             plot_density=True)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()




def make_cp_density_estimation_01(flow, iteration, labels, test_label):
  
    num_samples = 200

    samples = flow.sample(num_samples).squeeze().cpu().detach().numpy()

    for i in range(0, 50):
        samples_temp = flow.sample(num_samples).squeeze().cpu().detach().numpy()
        samples = np.vstack([samples, samples_temp])

    corner.corner(samples, # fig = fig
             labels=labels,
             show_titles=True,
             plot_datapoints=True,
             fill_contours=True, 
             bins=50,
             quantiles=[0.68, 0.954, 0.997],
             color='blue',
             weights=np.ones(samples.shape[0])/samples.shape[0],
             plot_density=True)
    plt.savefig('../experiments/plots/corner_' + str(test_label) + '_'+str(iteration)+'.png')
    plt.close()





def waveform_from_posterior():

    print('Not implemented')

    # Read in the parameters that are sampled from thebase but then projected on the posterior

    # Create a waveform for each set of parameters

    # Plot all the waveforms together to see their spread

    # Plot also the results of the embedding to see how the compression works
   





