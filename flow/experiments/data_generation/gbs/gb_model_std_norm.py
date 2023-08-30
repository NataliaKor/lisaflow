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

from flow.experiments.data_generation.base import Source

import lisabeta.lisa.pyLISAnoise as pyLISAnoise
from flow.utils.noisemodel import *
from flow.utils.transform_to_as import *

# Making code agnostic to CPU/GPU
def std_get_wrapper(arg):
    return arg

def cuda_get_wrapper(arg):
    return arg.get()

if torch.cuda.is_available():
   import cupy as xp
   gpu = True
   get_wrapper = cuda_get_wrapper
   print('gpu = ', gpu)
else:
   import numpy as xp
   gpu = False
   get_wrapper = std_get_wrapper
   print('gpu = ', gpu)

# Create samples of the noise with the defined variance
def sample_noise(variance, df, dt):#, sample_shape):

   #n_real = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   #n_imag = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   n_real = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(dt*xp.sqrt(4.0*df)))#, size=sample_shape)
   n_imag = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(dt*xp.sqrt(4.0*df)))#, size=sample_shape)

   return n_real+1j*n_imag
   
def sample_noise_amplitude(variance, df, dt):
 
   n_amp = xp.random.rayleigh(scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   n_phase = xp.random.uniform(low=0.0, high=2.0*xp.pi, size=variance.shape)

   return n_amp*xp.exp(1j*n_phase)
 

class GB_gpu(Source):
    def __init__(self, config, config_data, dtype):
         super().__init__()
         self.config = config
         self.config_data = config_data
  
         self.Tobs = config_data['tvec']['Tobs']
         self.dt = config_data['tvec']['dt']
      
         self.freqs = None
         self.df = None   

         self.num = None
         self.kmin = None  

         self.params0 = None # This is set sampled from the physical parameters, just for comparison
         self.params1 = None
         self.params2 = None

         self.param_batch = None

         self.dtype = dtype
         self.sample_physical = True

    # Sample parameters that are going to be learned from the prior.
    # Now we sample all parameters all the time from the prior!
    # We will marginalise over the ones that we do not learn. 
    def sample_from_prior(self, N, iteration):
    
        check_prior_on_as = False
       
        # Intrinsic parameters
        f0 =   xp.random.uniform(self.config_data['limits']['min']['f0'], self.config_data['limits']['max']['f0'], N) # Sample straight away in cupy
        fdot = xp.random.uniform(self.config_data['limits']['min']['fdot'], self.config_data['limits']['max']['fdot'], N)
            
        beta_sin = xp.random.uniform(self.config_data['limits']['min']['beta_sin'], self.config_data['limits']['max']['beta_sin'], N)
        beta = xp.arcsin(beta_sin)

        lam = xp.random.uniform(self.config_data['limits']['min']['lam'], self.config_data['limits']['max']['lam'] * xp.pi, N) # np.pi

        if self.sample_physical == True:

            amp = xp.random.uniform(self.config_data['limits']['min']['amp'], self.config_data['limits']['max']['amp'], N)

            iota_cos = xp.random.uniform(self.config_data['limits']['min']['iota_cos'], self.config_data['limits']['max']['iota_cos'], N)
            iota = xp.arccos(iota_cos)

            phi0 = xp.random.uniform(self.config_data['limits']['min']['phi0'], self.config_data['limits']['max']['phi0'] * xp.pi, N)  # np.pi
            psi =  xp.random.uniform(self.config_data['limits']['min']['psi'] * xp.pi, self.config_data['limits']['max']['psi'] * xp.pi, N)    # np.pi
    
            ffdot = xp.zeros(N) 
            self.params0 = xp.array([amp, f0, fdot, ffdot, phi0, iota, psi, lam, beta]) # Check if it matters if it is phi0 or -phi0
            # Construct batch of paramaters fro sampling
            sampling_parameters = xp.vstack((f0, fdot, beta_sin, lam, iota_cos, amp, phi0, psi)).T

        else:

            if check_prior_on_as:
                # Extrinsic parameters sampled in a's. Sample from the same prior range.
                amp = xp.random.uniform(self.config_data['limits']['min']['amp'], self.config_data['limits']['max']['amp'], N)

                iota_cos = xp.random.uniform(self.config_data['limits']['min']['iota_cos'], self.config_data['limits']['max']['iota_cos'], N)
                iota = xp.arccos(iota_cos)

                phi0 = xp.random.uniform(self.config_data['limits']['min']['phi0'], self.config_data['limits']['max']['phi0'] * xp.pi, N)  # np.pi
                psi =  xp.random.uniform(self.config_data['limits']['min']['psi'] * xp.pi, self.config_data['limits']['max']['psi'] * xp.pi, N)    # np.pi
    
                ffdot = xp.zeros(N) 
                self.params0 = xp.array([amp, f0, fdot, ffdot, -phi0, iota, psi, lam, beta])

                # We do not use this transform here. This is just to check what are the priors
                a1, a2, a3, a4 = transform_params(torch.as_tensor(amp).type(self.dtype), \
                                                  torch.as_tensor(iota_cos).type(self.dtype), \
                                                  torch.as_tensor(phi0).type(self.dtype), \
                                                  torch.as_tensor(psi).type(self.dtype))

                # Make a cormer plot of the parametres a to see their ranges
                figure = corner.corner(np.c_[a1.detach().cpu(), a2.detach().cpu(), a3.detach().cpu(), a4.detach().cpu()],
                              plot_datapoints = True,
                              labels=['a1','a2','a3', 'a4'],
                              show_titles=True)
                plt.savefig('corner_as.png') 
                plt.close()     
                a1 = xp.asarray(a1).squeeze()
                a2 = xp.asarray(a2).squeeze()
                a3 = xp.asarray(a3).squeeze()
                a4 = xp.asarray(a4).squeeze()
        
            # Sample parameters a's and transform them to the extrinsic parameters
            a1 = xp.random.uniform(self.config_data['limits']['min']['a_s'], self.config_data['limits']['max']['a_s'], N)
            a2 = xp.random.uniform(self.config_data['limits']['min']['a_s'], self.config_data['limits']['max']['a_s'], N)
            a3 = xp.random.uniform(self.config_data['limits']['min']['a_s'], self.config_data['limits']['max']['a_s'], N)
            a4 = xp.random.uniform(self.config_data['limits']['min']['a_s'], self.config_data['limits']['max']['a_s'], N)
 
            self.a_s = xp.array([a1, a2, a3, a4])
      
            # Parameters to make a linear combination of the waveforms
            #params = np.array([amp, f0, fdot, 0.0, -phi0, iota, psi, lam, beta])
            amp = 10**(-22)*xp.ones(N)
            ffdot = xp.zeros(N)
            iota = xp.arccos(xp.zeros(N))
            phi0 = xp.zeros(N)    
            psi1 = xp.zeros(N) 
            psi2 = 0.25*xp.pi*xp.ones(N)    
       
            self.params1 = xp.array([amp, f0, fdot, ffdot, -phi0, iota, psi1, lam, beta])
            self.params2 = xp.array([amp, f0, fdot, ffdot, -phi0, iota, psi2, lam, beta])

            # Construct batch of paramaters fro sampling
            sampling_parameters = xp.vstack((f0, fdot, beta_sin, lam, a1, a2, a3, a4)).T

        # Estimate this only when we run the first test batch of the data.
        print('iteration = ', iteration)
        print('resume = ', self.config['training']['resume'])
        if iteration == 0:
            if self.config['training']['resume'] == 0:
                # Standardise all parameters and record the parameters of the standardasion when it is done for the first time.
                print('Do we go here????????????????????????????')
                self.parameters_mean = np.mean(sampling_parameters, axis=0)
                print('self.parameters_mean = ', self.parameters_mean)
                self.parameters_std = np.std(sampling_parameters, axis=0)
                print('self.parameters_std = ', self.parameters_std)
                if self.sample_physical == True:
                    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'iota_cos', 'amp', 'phi0', 'psi']
                else:
                    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'a1', 'a2', 'a3', 'a4']

                np.savetxt('/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/means' + self.config['saving']['label'] + '.txt' , self.parameters_mean.get())
                np.savetxt('/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/stds' + self.config['saving']['label'] + '.txt', self.parameters_std.get())

            else:
                print('path to file:', '/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/means' + self.config['saving']['label'] + '.txt')
                self.parameters_mean = xp.loadtxt('/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/means' + self.config['saving']['label'] + '.txt')
                self.parameters_std = xp.loadtxt('/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/stds' + self.config['saving']['label'] + '.txt')
        
                if self.sample_physical == True:
                    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'iota_cos', 'amp', 'phi0', 'psi']
                else:
                    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'a1', 'a2', 'a3', 'a4']
        else:
            print('label = ', self.config['saving']['label'])
            self.parameters_mean = xp.loadtxt('/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/means' + self.config['saving']['label'] + '.txt')
            self.parameters_std = xp.loadtxt('/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments/stds' + self.config['saving']['label'] + '.txt')
            if self.sample_physical == True:
                self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'iota_cos', 'amp', 'phi0', 'psi']
            else:
                self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'a1', 'a2', 'a3', 'a4']
         
        # Return the set of standardised parameters to sample
        self.param_batch = (sampling_parameters - self.parameters_mean) / self.parameters_std

        #self.param_batch = torch.cat([param_batch, a1, f0_norm, fdot_norm, beta_norm, lam_norm, a2, a3, a4], dim=1)
         
    # Create waveform combinations
    def create_waveform(self):

        N_points = 128
        
        if self.sample_physical == True:
            gb1 = GBGPU(use_gpu=True)        
            gb1.run_wave(*self.params0, N = N_points, dt = self.dt, T = self.Tobs, oversample=2)
 
        else:

            gb1 = GBGPU(use_gpu=True)
            gb2 = GBGPU(use_gpu=True)
            gb1.run_wave(*self.params1, N = N_points, dt = self.dt, T = self.Tobs, oversample=2)
            gb2.run_wave(*self.params2, N = N_points, dt = self.dt, T = self.Tobs, oversample=2)


        self.df = 1./self.Tobs
      
        # Put waveforms in a common frequency band    
        self.k_min = np.round(self.config_data['limits']['min']['fvec']/self.df).astype(int)
        k_max = np.round(self.config_data['limits']['max']['fvec']/self.df).astype(int)

        self.num = k_max - self.k_min
      
        # This should be the same for both waveforms because intrinsic parameters are the same
        i_start = (gb1.start_inds.get() - self.k_min).astype(int)
        i_end = (gb1.start_inds.get() - self.k_min + gb1.N).astype(int)

        # Define frequency vector
        self.freqs = (xp.arange(self.num) + self.k_min)*self.df



        '''
          Add noise to the data and whiten frequency waveform with theoretical PSD.
        '''
        noise = AnalyticNoise(self.freqs, 'MRDv1')
        psd_A, psd_E = noise.psd(option="A"), noise.psd(option="E")

        noiseA = sample_noise(xp.array(psd_A), self.df, self.dt)
        noiseE = sample_noise(xp.array(psd_E), self.df, self.dt)
        
        batch_size = gb1.A.shape[0]

        if self.sample_physical == True:
            signal_A = gb1.A
            signal_E = gb1.E

        else:
            signal_A = self.a_s[0,:][:,None]*gb1.A  + 1.j*self.a_s[2,:][:,None]*gb1.A + self.a_s[1,:][:,None]*gb2.A + 1.j*self.a_s[3,:][:,None]*gb2.A
            signal_E = self.a_s[0,:][:,None]*gb1.E + 1.j*self.a_s[2,:][:,None]*gb1.E + self.a_s[1,:][:,None]*gb2.E + 1.j*self.a_s[3,:][:,None]*gb2.E
     
       # signal_A = signal_A
       # signal_E = signal_E

        # Compare the waveform constructed with F-statistics to the wavefomr constructed 
        plot_comparison = False
        if plot_comparison:
             plt.figure()
             plt.loglog(np.abs(gb0.A[0,:].get()))
             plt.loglog(np.abs(signal_A[0,:].get()))
             plt.savefig('comparison_A.png')
  

        A_white = xp.empty((batch_size, self.num), dtype=xp.complex128)
        E_white = xp.empty((batch_size, self.num), dtype=xp.complex128)

        # TIME IF THIS IS FASTER THAN PREVIOUS SOLUTION
      

        for i in range(batch_size):
            xA = xp.zeros(self.num, dtype = xp.complex128)
            xE = xp.zeros(self.num, dtype = xp.complex128)
            xA[i_start[i]:i_end[i]] = signal_A[i,:]
            xE[i_start[i]:i_end[i]] = signal_E[i,:]

            #xA = (xA/self.dt + noiseA)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_A))  # check if we need factor of dt here of not
            #xE = (xE/self.dt + noiseE)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_E))  # check if we need factor of dt here of not

            xA = (xA + noiseA)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_A))  # check if we need factor of dt here of not
            xE = (xE + noiseE)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_E))  # check if we need factor of dt here of not

            A_white[i,:] = xA        
            E_white[i,:] = xE
    
        return A_white, E_white


    # Create waveform combinations
    def create_waveform_nonoise(self):

        N_points = 128
        
        if self.sample_physical == True:
            gb1 = GBGPU(use_gpu=True)        
            gb1.run_wave(*self.params0, N = N_points, dt = self.dt, T = self.Tobs, oversample=2)
 
        else:

            gb1 = GBGPU(use_gpu=True)
            gb2 = GBGPU(use_gpu=True)
            gb1.run_wave(*self.params1, N = N_points, dt = self.dt, T = self.Tobs, oversample=2)
            gb2.run_wave(*self.params2, N = N_points, dt = self.dt, T = self.Tobs, oversample=2)


        self.df = 1./self.Tobs
      
        # Put waveforms in a common frequency band    
        self.k_min = np.round(self.config_data['limits']['min']['fvec']/self.df).astype(int)
        k_max = np.round(self.config_data['limits']['max']['fvec']/self.df).astype(int)

        self.num = k_max - self.k_min
      
        # This should be the same for both waveforms because intrinsic parameters are the same
        i_start = (gb1.start_inds.get() - self.k_min).astype(int)
        i_end = (gb1.start_inds.get() - self.k_min + gb1.N).astype(int)

        # Define frequency vector
        self.freqs = (xp.arange(self.num) + self.k_min)*self.df



        '''
          Add noise to the data and whiten frequency waveform with theoretical PSD.
        '''
        noise = AnalyticNoise(self.freqs, 'MRDv1')
        psd_A, psd_E = noise.psd(option="A"), noise.psd(option="E")

        noiseA = sample_noise(xp.array(psd_A), self.df, self.dt)
        noiseE = sample_noise(xp.array(psd_E), self.df, self.dt)
        
        batch_size = gb1.A.shape[0]

        if self.sample_physical == True:
            signal_A = gb1.A
            signal_E = gb1.E

        else:
            signal_A = self.a_s[0,:][:,None]*gb1.A  + 1.j*self.a_s[2,:][:,None]*gb1.A + self.a_s[1,:][:,None]*gb2.A + 1.j*self.a_s[3,:][:,None]*gb2.A
            signal_E = self.a_s[0,:][:,None]*gb1.E + 1.j*self.a_s[2,:][:,None]*gb1.E + self.a_s[1,:][:,None]*gb2.E + 1.j*self.a_s[3,:][:,None]*gb2.E
     
       # signal_A = signal_A
       # signal_E = signal_E

        # Compare the waveform constructed with F-statistics to the wavefomr constructed 
        plot_comparison = False
        if plot_comparison:
             plt.figure()
             plt.loglog(np.abs(gb0.A[0,:].get()))
             plt.loglog(np.abs(signal_A[0,:].get()))
             plt.savefig('comparison_A.png')
  

        A_white = xp.empty((batch_size, self.num), dtype=xp.complex128)
        E_white = xp.empty((batch_size, self.num), dtype=xp.complex128)

        # TIME IF THIS IS FASTER THAN PREVIOUS SOLUTION
      

        for i in range(batch_size):
            xA = xp.zeros(self.num, dtype = xp.complex128)
            xE = xp.zeros(self.num, dtype = xp.complex128)
            xA[i_start[i]:i_end[i]] = signal_A[i,:]
            xE[i_start[i]:i_end[i]] = signal_E[i,:]

            #xA = (xA/self.dt + noiseA)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_A))  # check if we need factor of dt here of not
            #xE = (xE/self.dt + noiseE)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_E))  # check if we need factor of dt here of not

            xA = (xA)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_A))  # check if we need factor of dt here of not
            xE = (xE)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_E))  # check if we need factor of dt here of not

            A_white[i,:] = xA        
            E_white[i,:] = xE
    
        return A_white, E_white




    def set_wf_params(self, params):
        self.params0 = params
 
    def get_params(self):
        return self.param_batch

    def get_param_label(self):
        return self.parameter_labels

    def get_param_mean(self):
        return self.parameters_mean

    def get_param_std(self):
        return self.parameters_std

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

        if self.sample_physical == True:
            truths = np.array([f0, fdot, np.sin(beta), lam, np.cos(iota), amp, phi0, psi])

        else:
            a1, a2, a3, a4 = transform_params(torch.as_tensor(amp).type(self.dtype), \
                                              torch.as_tensor(np.cos(iota)).type(self.dtype), \
                                              torch.as_tensor(phi0).type(self.dtype), \
                                              torch.as_tensor(psi).type(self.dtype))

            truths = np.array([f0, fdot, np.sin(beta), lam, a1.cpu().detach().numpy(), a2.cpu().detach().numpy(), a3.cpu().detach().numpy(), a4.cpu().detach().numpy()])
        N_points = 128

        # Create waveform
        gb = GBGPU(use_gpu=True)
        gb.run_wave(*params, N = N_points, dt = self.dt, T = self.Tobs, oversample = 2)#oversample=2)

        A_out = xp.zeros((1, self.num), dtype=xp.complex128)
        E_out = xp.zeros((1, self.num), dtype=xp.complex128)

        i_start = (gb.start_inds.get() - self.k_min).astype(np.int32)
        i_end = (gb.start_inds.get() - self.k_min + gb.N).astype(np.int32)

        A_out[0, i_start[0] : i_end[0]] = gb.A
        E_out[0, i_start[0] : i_end[0]] = gb.E
        
        print('A_out = ', A_out)
        print('E_out = ', E_out)

        noise = AnalyticNoise(self.freqs, 'MRDv1')
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
     
        noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)#, E_out.shape)

        #A_white = (A_out/self.dt + noiseA) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        #E_white = (E_out/self.dt + noiseE) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))

        A_white = (A_out + noiseA) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        E_white = (E_out + noiseE) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))
 
        # Have to check if this thing and stuff that goes as input to the network is the same
  
        return A_white, E_white, truths


    def true_data_nonoise(self):
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

        if self.sample_physical == True:
            truths = np.array([f0, fdot, np.sin(beta), lam, np.cos(iota), amp, phi0, psi])

        else:
            a1, a2, a3, a4 = transform_params(torch.as_tensor(amp).type(self.dtype), \
                                              torch.as_tensor(np.cos(iota)).type(self.dtype), \
                                              torch.as_tensor(phi0).type(self.dtype), \
                                              torch.as_tensor(psi).type(self.dtype))

            truths = np.array([f0, fdot, np.sin(beta), lam, a1.cpu().detach().numpy(), a2.cpu().detach().numpy(), a3.cpu().detach().numpy(), a4.cpu().detach().numpy()])
        N_points = 128

        # Create waveform
        gb = GBGPU(use_gpu=True)
        gb.run_wave(*params, N = N_points, dt = self.dt, T = self.Tobs, oversample = 2)#oversample=2)

        A_out = xp.zeros((1, self.num), dtype=xp.complex128)
        E_out = xp.zeros((1, self.num), dtype=xp.complex128)

        i_start = (gb.start_inds.get() - self.k_min).astype(np.int32)
        i_end = (gb.start_inds.get() - self.k_min + gb.N).astype(np.int32)

        A_out[0, i_start[0] : i_end[0]] = gb.A
        E_out[0, i_start[0] : i_end[0]] = gb.E
        
        print('A_out = ', A_out)
        print('E_out = ', E_out)

        noise = AnalyticNoise(self.freqs, 'MRDv1')
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
     
        noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)#, E_out.shape)

        #A_white = (A_out/self.dt + noiseA) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        #E_white = (E_out/self.dt + noiseE) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))

        A_white = (A_out) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        E_white = (E_out) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))
 
        # Have to check if this thing and stuff that goes as input to the network is the same
  
        return A_white, E_white, truths



    def check_waveforms(self, beta, lam, index):
        '''
          "True" data for testing. Dataset with the default parameters.
        '''
        amp = self.config_data['default']['amp']
        f0 = self.config_data['default']['f0']
        fdot = self.config_data['default']['fdot']

        phi0 = self.config_data['default']['phi0']
        iota = self.config_data['default']['iota']
        psi = self.config_data['default']['psi']
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! - phi0
        params = np.array([amp, f0, fdot, 0.0, -phi0, iota, psi, lam, beta])

        a1, a2, a3, a4 = transform_params(torch.as_tensor(amp).type(self.dtype), \
                                          torch.as_tensor(np.cos(iota)).type(self.dtype), \
                                          torch.as_tensor(phi0).type(self.dtype), \
                                          torch.as_tensor(psi).type(self.dtype))

        print('type(beta) = ', type(beta))
        print('type(lam) = ', type(lam))
        truths = np.array([f0, fdot, np.sin(beta), lam, a1.cpu().detach().numpy(), a2.cpu().detach().numpy(), a3.cpu().detach().numpy(), a4.cpu().detach().numpy()], dtype=np.float64)
        N_points = 128

        # Create waveform
        gb = GBGPU(use_gpu=True)
        gb.run_wave(*params, N = N_points, dt = self.dt, T = self.Tobs, oversample = 2)#oversample=2)

        A_out = xp.zeros((1, self.num), dtype=xp.complex128)
        E_out = xp.zeros((1, self.num), dtype=xp.complex128)

        i_start = (gb.start_inds.get() - self.k_min).astype(np.int32)
        i_end = (gb.start_inds.get() - self.k_min + gb.N).astype(np.int32)

        A_out[0, i_start[0] : i_end[0]] = gb.A
        E_out[0, i_start[0] : i_end[0]] =  gb.E

        noise = AnalyticNoise(self.freqs, 'MRDv1')
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")

        noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)#, E_out.shape)

        A_white = (A_out + noiseA) * xp.sqrt(4.0*self.df)*self.dt*self.dt/xp.sqrt(xp.array(noisevals_A)) # !!!! TWO TIMES MULTIPLY BY dt
        E_white = (E_out + noiseE) * xp.sqrt(4.0*self.df)*self.dt*self.dt/xp.sqrt(xp.array(noisevals_E))  # !!!! TWO TIMES MULTIPLY BY dt
     
        # Have to check if this thing and stuff that goes as input to the network is the same
        #plt.figure()
        #plt.loglog(np.abs(A_white[0,:].get()))
        #plt.savefig('A_' + str(index) + '.png') 
        #plt.close()

        return A_white, E_white, truths

