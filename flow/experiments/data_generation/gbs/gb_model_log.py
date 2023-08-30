'''
  Generation of the GB waveforms on the fly.
'''
import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
import corner
import torch

from gbgpu.gbgpu import GBGPU
from gbgpu.utils.constants import *

from data_generation.base import Source

import lisabeta.lisa.pyLISAnoise as pyLISAnoise
from flow.utils.noisemodel import *
from flow.utils.transform_to_as import *

# Create samples of the noise with the defined variance
def sample_noise(variance, df, dt):#, sample_shape):

    #print('np.sqrt(variance) = ', np.sqrt(variance))
   n_real = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   n_imag = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)

   return n_real+1j*n_imag
   
def sample_noise_amplitude(variance, df, dt):
 
   n_amp = xp.random.rayleigh(scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   n_phase = xp.random.uniform(low=0.0, high=2.0*xp.pi, size=variance.shape)

   return n_amp*xp.exp(1j*n_phase)
 
# Normalise parameters to be from 0 to 1
def normalise(par, par_min, par_max):

    return (par - par_min)/(par_max - par_min)


class GB_gpu(Source):
    def __init__(self, config, config_data, dtype):
         super().__init__()
         self.config = config
         self.config_data = config_data
         self.gb = None

         self.Tobs = config_data['tvec']['Tobs']
         self.dt = config_data['tvec']['dt']
      
         self.freqs = None
         self.df = None   
         self.A = None
         self.E = None   

         self.num = None
         self.kmin = None  

         self.dtype = dtype


    # TODO replace it as a loop, for that I have to be able to import constants in the config file
    # Return parameter ranges and labels for plotting and rescating
    def param_ranges(self):

        # Define values for fixed parameters
        # and boundaries for varied parameters
        parameter_labels = []
        param_min = []
        param_max = []
        params = []

        if self.config_data['estimate']['amp']:

            param_min.append(0.0)
            param_max.append(1.0) 

            #param_min.append(self.config_data['limits']['min']['amp'])
            #param_max.append(self.config_data['limits']['max']['amp'])
            parameter_labels.append('amp')
            #params.append(self.config_data['default']['amp']/(10**self.config_data['limits']['max']['amp']))

            params.append(self.config_data['default']['amp'])



        if self.config_data['estimate']['f0']:

            param_min.append(self.config_data['limits']['min']['f0'])
            param_max.append(self.config_data['limits']['max']['f0'])
            parameter_labels.append('f0')
            params.append(np.log10(self.config_data['default']['f0']))

        if self.config_data['estimate']['fdot']:

            param_min.append(self.config_data['limits']['min']['fdot'])
            param_max.append(self.config_data['limits']['max']['fdot'])
            parameter_labels.append('fdot')
            params.append(np.log10(self.config_data['default']['fdot']))

        if self.config_data['estimate']['beta']:

            param_min.append(self.config_data['limits']['min']['beta_sin'])
            param_max.append(self.config_data['limits']['max']['beta_sin']) 
            parameter_labels.append('sbeta')
            params.append(np.sin(self.config_data['default']['beta']))


        if self.config_data['estimate']['lam']:

            param_min.append(self.config_data['limits']['min']['lam'])
            param_max.append(self.config_data['limits']['max']['lam'] * np.pi)
            parameter_labels.append('lam')
            params.append(self.config_data['default']['lam']) 

        if self.config_data['estimate']['iota']:

            param_min.append(0.0)
            param_max.append(1.0)
            #param_min.append(self.config_data['limits']['min']['iota_cos'])
            #param_max.append(self.config_data['limits']['max']['iota_cos'])
            #param_max.append(self.config_data['limits']['max']['iota'] * np.pi)
            parameter_labels.append('ciota')
            params.append(np.cos(self.config_data['default']['iota']))

        if self.config_data['estimate']['psi']:

            param_min.append(0.0)
            param_max.append(1.0) 
            #param_min.append(self.config_data['limits']['min']['psi'])
            #param_max.append(self.config_data['limits']['max']['psi'] * np.pi)
            parameter_labels.append('psi')
            params.append(self.config_data['default']['psi'])

        if self.config_data['estimate']['phi0']:

            param_min.append(0.0)
            param_max.append(1.0) 
            #param_min.append(self.config_data['limits']['min']['phi0'])
            #param_max.append(self.config_data['limits']['max']['phi0'] * np.pi)
            parameter_labels.append('phi0')
            params.append(self.config_data['default']['phi0'])

        return param_min, param_max, parameter_labels, params


    def freqwave_AET(self, N):

        param_batch = torch.Tensor().type(self.dtype)

        # Read parameters from the configuration file
        if self.config_data['estimate']['amp']:
            amp_log = np.random.uniform(self.config_data['limits']['min']['amp'], self.config_data['limits']['max']['amp'], N)
            amp = 10.0**(amp_log) 
            amp_norm = torch.from_numpy(amp).type(self.dtype).view(-1,1)/(10**self.config_data['limits']['max']['amp'])/2
        else:
            amp = np.full((N), self.config_data['default']['amp'])

        if self.config_data['estimate']['f0']:
            f0_log = np.random.uniform(self.config_data['limits']['min']['f0'], self.config_data['limits']['max']['f0'], N)
            f0_norm = torch.from_numpy(normalise(f0_log, self.config_data['limits']['min']['f0'], self.config_data['limits']['max']['f0'])).type(self.dtype).view(-1,1)
            f0 = 10.0**f0_log
            #param_batch = torch.cat([param_batch, f0_norm], dim=1) 
        else:
            f0 = np.full((N), self.config_data['default']['f0'])

        if self.config_data['estimate']['fdot']:
            fdot_log = np.random.uniform(self.config_data['limits']['min']['fdot'], self.config_data['limits']['max']['fdot'], N)
            fdot_norm = torch.from_numpy(normalise(fdot_log, self.config_data['limits']['min']['fdot'], self.config_data['limits']['max']['fdot'])).type(self.dtype).view(-1,1)
            fdot = 10.0**fdot_log
            #param_batch = torch.cat([param_batch, fdot_norm], dim=1) 
        else:
            fdot = np.full((N), self.config_data['default']['fdot'])

        if self.config_data['estimate']['beta']:
            beta_sin = np.random.uniform(self.config_data['limits']['min']['beta_sin'], self.config_data['limits']['max']['beta_sin'], N)
            beta = np.arcsin(beta_sin)
            beta_norm = torch.from_numpy(normalise(beta_sin, self.config_data['limits']['min']['beta_sin'], self.config_data['limits']['max']['beta_sin'])).type(self.dtype).view(-1,1) 
            #param_batch = torch.cat([param_batch, beta_norm], dim=1) 
        else:
            beta = np.full((N), self.config_data['default']['beta'])

        if self.config_data['estimate']['lam']:
            lam = np.random.uniform(self.config_data['limits']['min']['lam'], self.config_data['limits']['max']['lam'] * np.pi, N)
            lam_norm = torch.from_numpy(normalise(lam, self.config_data['limits']['min']['lam'], self.config_data['limits']['max']['lam'] * np.pi)).type(self.dtype).view(-1,1)
            #param_batch = torch.cat([param_batch, lam_norm], dim=1) 
        else:
            lam = np.full((N), self.config_data['default']['lam'])

        if self.config_data['estimate']['iota']:
            iota_cos = np.random.uniform(self.config_data['limits']['min']['iota_cos'], self.config_data['limits']['max']['iota_cos'], N)
            iota = np.arccos(iota_cos)
        else:
            #iota_norm = torch.from_numpy(normalise(iota_cos, self.config_data['limits']['min']['iota_cos'], self.config_data['limits']['max']['iota_cos'] )).type(self.dtype).view(-1,1)
            #param_batch = torch.cat([param_batch, iota_norm], dim=1) 
            iota = np.full((N), self.config_data['default']['iota'])

        if self.config_data['estimate']['psi']:
            psi = np.random.uniform(self.config_data['limits']['min']['psi'], self.config_data['limits']['max']['psi'] * np.pi, N)
            #psi_norm = torch.from_numpy(normalise(psi, self.config_data['limits']['min']['psi'], self.config_data['limits']['max']['psi'] * np.pi)).type(self.dtype).view(-1,1)
            #param_batch = torch.cat([param_batch, psi_norm], dim=1)
        else:
            psi = np.full((N), self.config_data['default']['psi'])

        if self.config_data['estimate']['phi0']:
            phi0 = np.random.uniform(self.config_data['limits']['min']['phi0'], self.config_data['limits']['max']['phi0'] * np.pi, N)
            #phi0_norm = torch.from_numpy(normalise(phi0, self.config_data['limits']['min']['phi0'], self.config_data['limits']['max']['phi0'] * np.pi)).type(self.dtype).view(-1,1)
            #param_batch = torch.cat([param_batch, phi0_norm], dim=1) 
        else:
            phi0 = np.full((N), self.config_data['default']['phi0'])



       # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -phi0
        params = np.array([amp, f0, fdot, 0.0, -phi0, iota, psi, lam, beta])

        # TRANSFORMATION
        a1, a2, a3, a4 = transform_params(amp_norm.squeeze(), \
                                          torch.from_numpy(iota_cos).type(self.dtype), \
                                          torch.from_numpy(phi0).type(self.dtype), \
                                          torch.from_numpy(psi).type(self.dtype))       



        # Make a cormer plot of the parametres a to see their ranges
        #figure = corner.corner(np.array([a1.detach().cpu(), a2.detach().cpu(), a3.detach().cpu(), a4.detach().cpu()]),
        #              labels=['a1','a2','a3', 'a4'],
        #              show_titles=True)
        #
        #plt.close()


        # Combine the batch that we are going to sample
        #self.param_batch = torch.cat([param_batch, a1, f0_norm, fdot_norm, beta_norm, lam_norm, a2, a3, a4], dim=1)
        self.param_batch = torch.cat([param_batch, a1, f0_norm, fdot_norm, beta_norm, lam_norm, a2, a3, a4], dim=1)
       
        # Transformation back
        #Amp, cosinc, phi0_new, psi_new = reconstruct_params(a1, a2, a3, a4)

        # Rewrite set of the parameters that will be passed to the batch
        #self.param_batch = torch.cat([param_batch, iota_norm], dim=1)
 
        #figure = corner.corner(a2plot.transpose(),
        #                       labels=['a1','a2','a3','a4'],
        #                       show_titles=True)
        #                       plt.savefig('as.png')

       
        # number of points in waveform
        # if None, will determine inside the code based on amp, f0 (and P2 if running third-body waveform)
        N_points = None

        # Create waveform
        gb = GBGPU(use_gpu=True)
        gb.run_wave(*params, N = N_points, dt = self.dt, T = self.Tobs, oversample = 1)#oversample=2)
        self.gb = gb    
        batch_size = gb.A.shape[0]
        
        self.df = 1./self.Tobs   

        # Put waveforms in a common frequency band    
        self.k_min = np.round(self.config_data['limits']['min']['fvec']/self.df).astype(int)
        k_max = np.round(self.config_data['limits']['max']['fvec']/self.df).astype(int)
       
        self.num = k_max - self.k_min

        self.freqs = (np.arange(self.num) + self.k_min)*self.df
       
        A_out = xp.zeros((batch_size, self.num), dtype=xp.complex128)
        E_out = xp.zeros((batch_size, self.num), dtype=xp.complex128)

        i_start = (gb.start_inds.get() - self.k_min).astype(np.int32)
        i_end = (gb.start_inds.get() - self.k_min + gb.N).astype(np.int32)
       
        # THRES HAS TO BE A WAY TO DO IT WITHOUT THE LOOP!!!
        for i in range(batch_size):
            A_out[i, i_start[i] : i_end[i]] = gb.A[i]
            E_out[i, i_start[i] : i_end[i]] =  gb.E[i]

        '''
          Add noise to the data and whiten frequency waveform with theoretical PSD.
        ''' 
        noise = AnalyticNoise(self.freqs)
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
      
        noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)#, E_out.shape)

        #noise_A_ampph = sample_noise_amplitude(xp.array(noisevals_A), self.df, self.dt)

        #plt.figure()
        #plt.plot(noiseA.get())
        #plt.plot(noise_A_ampph.get())
        #plt.savefig('noise.png')

        #exit()

        A_noise = A_out + noiseA
        E_noise = E_out + noiseE

        A_white = A_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        E_white = E_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))

        return A_white, E_white  

    def get_params(self):
        return self.param_batch


    def get_freqs(self):
        return self.freqs 


    def true_data(self):
        '''
          "True" data for testing. Dataset with the default parameters.
        '''
        amp   = self.config_data['default']['amp']
        f0 = self.config_data['default']['f0']
        fdot = self.config_data['default']['fdot']
       
        phi0 = self.config_data['default']['phi0']
        iota = self.config_data['default']['iota']
        psi = self.config_data['default']['psi'] 
        lam = self.config_data['default']['lam']
        beta = self.config_data['default']['beta']
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! - phi0
        params = np.array([amp, f0, fdot, 0.0, -phi0, iota, psi, lam, beta])
        N_points = None

        # Create waveform
        gb = GBGPU(use_gpu=True)
        gb.run_wave(*params, N = N_points, dt = self.dt, T = self.Tobs, oversample = 1)#oversample=2)

        A_out = xp.zeros((1, self.num), dtype=xp.complex128)
        E_out = xp.zeros((1, self.num), dtype=xp.complex128)

        i_start = (gb.start_inds.get() - self.k_min).astype(np.int32)
        i_end = (gb.start_inds.get() - self.k_min + gb.N).astype(np.int32)

        A_out[0, i_start[0] : i_end[0]] = gb.A
        E_out[0, i_start[0] : i_end[0]] =  gb.E
        
        noise = AnalyticNoise(self.freqs)
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
     
        noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)#, E_out.shape)

        A_noise = A_out + noiseA
        E_noise = E_out + noiseE
  
        A_white = A_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        E_white = E_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))
 
        return A_white, E_white
