import numpy as np 
import cupy as cp
import torch
import corner
from matplotlib import pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_theme(style="whitegrid")

from flow.experiments.data_generation.gbs.gb_model_std_norm import GB_gpu
from flow.experiments.flow_architecture_play import *

from flow.distributions.normal import *
from flow.distributions.resample import *
from flow.utils.torchutils import *

from flow.networks.mlp import MLP
from flow.networks.resnet import ConvResidualNet

# Take a sample from the posteriors, calculate waveform
def sample_from_base(flow, coeff_norm, truths, param_mean, param_std, labels, label_plot, num_samples, num_repeat):

    samples = flow.sample(num_samples, coeff_norm).squeeze().cpu().detach().numpy()

    for i in range(0, num_repeat):
        samples_temp = flow.sample(num_samples, coeff_norm).squeeze().cpu().detach().numpy()
        samples = np.vstack([samples, samples_temp])

    for j in range(param_mean.shape[0]):
        samples[:,j] = samples[:,j]*param_std[j] + param_mean[j]
 
    figure = corner.corner(samples,
             labels=labels,
             show_titles=True, truths=truths)
    plt.savefig('../experiments/plot_results/corner_' + str(label_plot) + '.png')
    plt.close()

    return samples



# Calculate likelihood
def likelihood(d, h):
    
    d_h = d - h
    d_r = np.real(d)
    d_i = np.imag(d)
    h_r = np.real(h)
    h_i = np.imag(h)
    
    return  -np.sum((d_r*d_r + d_i*d_i + h_r*h_r + h_i*h_i - 2.*h_r*d_r - 2.*h_i*d_i)/2., axis=1)
    #return -d_h*np.conj(d_h)/2.0
    


# Calculate embedding
#def embedding():





def main():

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
    config = get_config('../experiments/configs/gbs/gb_resample.yaml')
    config_data = get_config('../experiments/configs/gbs/gb_as.yaml')

    # Load values for mean and variance
    print('label = ', config['saving']['label'])
    print('pathto file: ', '../experiments/means' + config['saving']['label'] + '.txt')
    #param_mean = np.loadtxt('../experiments/means' + config['saving']['label'] + '.txt')
    #param_std = np.loadtxt('../experiments/stds' + config['saving']['label'] + '.txt')

    #print('param_mean = ', param_mean)
    #print('param_std = ', param_std)

    num_samples = 2000
    num_repeat = 100
    label_plot = 'test_'

    # Size of the physical parameters
    features_size = config['model']['base']['params']

    # Define base distribution.
    if config['model']['base']['gaussian']:
        distribution = StandardNormal((features_size,)).to(dev)
    else:
        acceptance_fn = MLP(
        in_shape = [features_size],
        out_shape = [1],
        hidden_sizes = [512, 512],
        activation = F.leaky_relu,
        activate_output = True,
        activation_output = torch.sigmoid)
        distribution = ResampledGaussian((features_size,), acceptance_fn).to(dev)

    transform = create_transform(config).to(dev)

    # Define embedding network
    embed_in = 5 # 4
    embedding_net = ConvResidualNet(
            in_channels = embed_in,
            out_channels = 1,
            hidden_channels = config['model']['embedding']['hidden_features'],
            context_channels = None,
            num_blocks = config['model']['embedding']['num_blocks'],
            dropout_probability = config['model']['embedding']['dropout'],
            use_batch_norm = config['model']['embedding']['batch_norm'],
    )
 

    # Initialise network
    flow = Flow(transform, distribution, embedding_net).to(dev)

    # Load checkpoint
    print(config['training']['checkpoints'])
   
    checkpoint = torch.load(config['saving']['save_root'] + config['training']['checkpoints'])
    flow.load_state_dict(checkpoint['model_state_dict'])

    flow.eval()
    with torch.no_grad():

        # Create "real" data    
        max_batch = 100.0
        gb = GB_gpu(config, config_data, dtype)
        gb.sample_from_prior(1, 1)
        Atemp, Etemp = gb.create_waveform()
        A, E, truths = gb.true_data()
        print('A = ', A)
        print('E = ', E)
        print('truths = ', truths)
   
        param_mean = gb.get_param_mean().get()
        param_std = gb.get_param_std().get()

        freqs = (gb.get_freqs() - param_mean[0]) / param_std[0]
        print('freqs = ', freqs)
        freqs_one = torch.as_tensor(freqs).view(1,-1).type(dtype).view(1, 1, -1)
        freqs_arr = torch.as_tensor(cp.tile(freqs,(num_samples, 1))).type(dtype).view(num_samples, 1, -1)

        labels = gb.get_param_label()

   
        waveform = torch.cat((torch.as_tensor(np.real(A)/max_batch).type(dtype).view(1, 1, -1),
                          torch.as_tensor(np.real(E)/max_batch).type(dtype).view(1, 1, -1),
                          torch.as_tensor(np.imag(A)/max_batch).type(dtype).view(1, 1, -1),
                          torch.as_tensor(np.imag(E)/max_batch).type(dtype).view(1, 1, -1), freqs_one), 1)

        waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))


        # Produce samples
        sample_from_base(flow, waveform_cnn, truths, param_mean, param_std, labels, label_plot, num_samples, num_repeat)

        # Create many waveforms for samples and overplot them
        samples_for_plot = flow.sample(num_samples, waveform_cnn).squeeze().cpu().detach().numpy()

        # Return samples back to the normal scale
        for j in range(param_mean.shape[0]):
            samples_for_plot[:,j] = samples_for_plot[:,j]*param_std[j] + param_mean[j]

        # Arrange them in a way that is correct for the creation of the waveform
        # ['f0', 'fdot', 'beta_sin', 'lam', 'iota_cos', 'amp', 'phi0', 'psi'] 
        f0 = samples_for_plot[:,0]
        fdot = samples_for_plot[:,1]
        beta = np.arcsin(samples_for_plot[:,2])
        lam = samples_for_plot[:,3]
        print('samples_for_plor[:,4] = ', samples_for_plot[:,4])
        iota = np.arccos(samples_for_plot[:,4])
        amp = samples_for_plot[:,5]
        phi0 = samples_for_plot[:,6]
        psi = samples_for_plot[:,7]
        ffdot = np.zeros(num_samples)       
 
        params_for_plot = np.array([amp, f0, fdot, ffdot, phi0, iota, psi, lam, beta]) 
      
        # Create a waveform
        gb.set_wf_params(params_for_plot)
        A_plot, E_plot = gb.create_waveform_nonoise()
        A_plot_noise, E_plot_noise = gb.create_waveform()

        # True likelihood value
        A_nonoise, E_nonoise, truths =  gb.true_data_nonoise()
        likel_values_true = likelihood(A, A_nonoise)
        print('True likelihood = ', likel_values_true)

        # Plot the values of the likelihoods
        plt.figure()
        for j in range(param_mean.shape[0]):
            plt.plot(np.abs(A_plot[j,420:550].get()))

        plt.savefig('A.png')
        plt.close()

        # Calculate likelihood values and plot them
        likel_values = likelihood(A, A_plot)  
      
        plt.figure()
        plt.plot(likel_values.get())
        plt.savefig('likelihood.png')
        plt.close()
     
        print('likel_values.shape = ', likel_values.shape)
        print('max likelihood value', np.amax(likel_values.get(),where=~np.isnan(likel_values.get()),initial=-1000000))
        index = np.argwhere(likel_values.get() > -20000)

        #g = sns.relplot(x=beta, y=lam,
        #                size=likel_values.get(), sizes=(10, 200))
        #g = sns.pairplot(params_for_plot, size=likel_values.get(), sizes=(20, 200))
        params_plotting = np.squeeze(params_for_plot[:,index].T)
        likel_plotting = likel_values.get()[index ] #np.expand_dims(likel_values.get()[index], axis=1)
        print('params_plotting.shape = ', params_plotting.shape)
        print('likel_plotting.shape = ', likel_plotting.shape)
        sample_plotting = pd.DataFrame(data = np.hstack([params_plotting, likel_plotting]), 
                                       columns= ['amp','f0','fdot','ffdot', 'phi0', 'iota', 'psi', 'lam', 'beta', 'likel']) 
        
        g = sns.PairGrid(sample_plotting, hue = 'likel', corner=True)
        g.map(sns.scatterplot)
        g.add_legend()
        #fig = g.get_figure()
        plt.savefig("out_index.png")

        waveform_plot = torch.cat((torch.as_tensor(np.real(A_plot_noise)/max_batch).type(dtype).view(num_samples, 1, -1),
                              torch.as_tensor(np.real(E_plot_noise)/max_batch).type(dtype).view(num_samples, 1, -1),
                              torch.as_tensor(np.imag(A_plot_noise)/max_batch).type(dtype).view(num_samples, 1, -1),
                              torch.as_tensor(np.imag(E_plot_noise)/max_batch).type(dtype).view(num_samples, 1, -1), freqs_arr), 1)

        waveform_cnn_plot = torch.reshape(waveform_plot, (waveform_plot.shape[0], embed_in, 1, -1))
 
   
        # Create many embeggings for samples and overplot them
        embeddings = embedding_net(waveform_cnn_plot)
        print('embeddings.shape = ', embeddings.shape)
        plt.figure()
        plt.plot(embeddings.cpu().detach().numpy().T)
        plt.savefig('embeddings.png')
        plt.close() 

if __name__ == '__main__':
    main()

