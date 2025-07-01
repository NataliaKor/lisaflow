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
def sample_noise(variance, df, dt, batch_size, num):

   #n_real = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   #n_imag = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   n_real = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(xp.sqrt(4.0*df)), size=(batch_size, num))
   n_imag = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(xp.sqrt(4.0*df)), size=(batch_size, num))

   return n_real+1j*n_imag
   
def sample_noise_amplitude(variance, df, dt):
 
   n_amp = xp.random.rayleigh(scale=np.sqrt(variance)/(dt*np.sqrt(4.0*df)))#, size=sample_shape)
   n_phase = xp.random.uniform(low=0.0, high=2.0*xp.pi, size=variance.shape)

   return n_amp*xp.exp(1j*n_phase)
 

class GB_gpu(Source):
    def __init__(self, config, config_data, f0, num_f, dtype):
         super().__init__()
         self.config = config
         self.config_data = config_data
  
         self.Tobs = config_data['tvec']['Tobs'] * YEAR
         self.dt = config_data['tvec']['dt']
         self.df = 1./self.Tobs
       
         self.f0 = f0 # true f0 of the signal, this is used to define the band and prior on f0
         self.num_f = num_f # number of frequencies defined for each GB, depending on f0 and SNR, i.e. how many frequencies around f0 we should take   

         self.freqs = None
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
        '''
          Num_f -- is the number of frequencies 
        '''
    
        check_prior_on_as = False

        sampling_parameters = xp.empty((N,0))
        self.parameter_labels = []
       
        # Intrinsic parameters
        if self.config_data['estimate']['f0']:
            f0_min = self.f0 - self.num_f*self.df
            f0_max = self.f0 + self.num_f*self.df
            f0 =   xp.random.uniform(f0_min, f0_max, N) # Sample straight away in cupy
            f0_dim = xp.expand_dims(f0, axis = 1)
            sampling_parameters = xp.append(sampling_parameters, f0_dim, axis = 1)
            self.parameter_labels.append('f0')
        else:
            f0 = xp.repeat(xp.array(self.config_data['default']['f0']),N)

        if self.config_data['estimate']['fdot']:
            fdot = xp.random.uniform(self.config_data['limits']['min']['fdot'], self.config_data['limits']['max']['fdot'], N)
            fdot_dim = xp.expand_dims(fdot, axis = 1)
            sampling_parameters = xp.append(sampling_parameters, fdot_dim, axis = 1)
            self.parameter_labels.append('fdot')
        else:
            fdot = xp.repeat(xp.array(self.config_data['default']['fdot']),N)          

        beta_sin = xp.random.uniform(self.config_data['limits']['min']['beta_sin'], self.config_data['limits']['max']['beta_sin'], N)
        beta = xp.arcsin(beta_sin)
        beta_sin_dim = xp.expand_dims(beta_sin,axis = 1)

        sampling_parameters = xp.append(sampling_parameters, beta_sin_dim, axis = 1)
        self.parameter_labels.append('sbeta')
 
        lam = xp.random.uniform(self.config_data['limits']['min']['lam'] * xp.pi , self.config_data['limits']['max']['lam'] * xp.pi, N) # np.pi
        lam_dim = xp.expand_dims(lam, axis = 1) 
        sampling_parameters = xp.append(sampling_parameters, lam_dim, axis = 1)
        self.parameter_labels.append('lam')
 

        if self.sample_physical == True:
            if self.config_data['estimate']['amp']:
                amp = xp.random.uniform(self.config_data['limits']['min']['amp'], self.config_data['limits']['max']['amp'], N)
                sampling_parameters = xp.vstack((sampling_parameters, amp))
                self.parameter_labels.append('amp') 
            else: 
                amp = xp.repeat(xp.array(self.config_data['default']['amp']),N)

            if self.config_data['estimate']['iota']:
                iota_cos = xp.random.uniform(self.config_data['limits']['min']['iota_cos'], self.config_data['limits']['max']['iota_cos'], N)
                iota = xp.arccos(iota_cos)
                sampling_parameters = xp.vstack((sampling_parameters, iota))
                self.parameter_labels.append('iota') 
            else:
                iota = xp.repeat(xp.array(self.config_data['default']['iota']),N)
                iota_cos = xp.cos(iota)

            if self.config_data['estimate']['phi0']: 
                phi0 = xp.random.uniform(self.config_data['limits']['min']['phi0'], self.config_data['limits']['max']['phi0'] * xp.pi, N)  # np.pi
                sampling_parameters = xp.vstack((sampling_parameters, phi0)) 
                self.parameter_labels.append('phi0') 
            else:
                phi0 = xp.repeat(xp.array(self.config_data['default']['phi0']),N)

            if self.config_data['estimate']['psi']:
                psi = xp.random.uniform(self.config_data['limits']['min']['psi'] * xp.pi, self.config_data['limits']['max']['psi'] * xp.pi, N)    # np.pi
                sampling_parameters = xp.vstack((sampling_parameters, psi))
                self.parameter_labels.append('psi')
            else:
                psi = xp.repeat(xp.array(self.config_data['default']['psi']),N) 
                
            ffdot = xp.zeros(N) 
            self.params0 = xp.array([amp, f0, fdot, ffdot, -phi0, iota, psi, lam, beta]) # Check if it matters if it is phi0 or -phi0
            # Construct batch of paramaters fro sampling

            #sampling_parameters = sampling_parameters.T
            #sampling_parameters = xp.vstack((f0, fdot, beta_sin, lam, iota_cos, amp, phi0, psi)).T

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
        if iteration == 0:
            if self.config['training']['resume'] == 0:
                # Standardise all parameters and record the parameters of the standardasion when it is done for the first time.
                self.parameters_mean = np.mean(sampling_parameters, axis=0)
                print('self.parameters_mean = ', self.parameters_mean)
                self.parameters_std = np.std(sampling_parameters, axis=0)
                print('self.parameters_std = ', self.parameters_std)
                #if self.sample_physical == True:
                #    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'iota_cos', 'amp', 'phi0', 'psi']
                #else:
                #    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'a1', 'a2', 'a3', 'a4']

                np.savetxt('means' + self.config['saving']['label'] + '.txt' , self.parameters_mean.get())
                np.savetxt('stds' + self.config['saving']['label'] + '.txt', self.parameters_std.get())

            else:
                self.parameters_mean = xp.loadtxt('means' + self.config['saving']['label'] + '.txt')
                self.parameters_std = xp.loadtxt('stds' + self.config['saving']['label'] + '.txt')
        
                #if self.sample_physical == True:
                #    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'iota_cos', 'amp', 'phi0', 'psi']
                #else:
                #    self.parameter_labels = ['f0', 'fdot', 'beta_sin', 'lam', 'a1', 'a2', 'a3', 'a4']

        # Return the set of standardised parameters to sample
        self.param_batch = (sampling_parameters - self.parameters_mean) / self.parameters_std

        #self.param_batch = torch.cat([param_batch, a1, f0_norm, fdot_norm, beta_norm, lam_norm, a2, a3, a4], dim=1)

         
    # Create waveform combinations
    def create_waveform(self, iteration):

 
        if self.sample_physical == True:
            gb1 = GBGPU(use_gpu=True)        
            gb1.run_wave(*self.params0, N = self.num_f, dt = self.dt, T = self.Tobs, oversample=2)
 
        else:

            gb1 = GBGPU(use_gpu=True)
            gb2 = GBGPU(use_gpu=True)
            gb1.run_wave(*self.params1, N = self.num_f, dt = self.dt, T = self.Tobs, oversample=2)
            gb2.run_wave(*self.params2, N = self.num_f, dt = self.dt, T = self.Tobs, oversample=2)

        # Put waveforms in a common frequency band  
        #print('self.num_f = ', self.num_f)  
        fvec_min = self.f0 - self.df*3.0*(self.num_f)/2.0
        fvec_max = self.f0 + self.df*3.0*(self.num_f)/2.0

        self.k_min = np.floor(fvec_min/self.df).astype(int)
        k_max = np.ceil(fvec_max/self.df).astype(int)

        self.num = k_max - self.k_min
      
        # This should be the same for both waveforms because intrinsic parameters are the same
        i_start = (gb1.start_inds.get() - self.k_min).astype(int)
        i_end = (gb1.start_inds.get() - self.k_min + gb1.N).astype(int)

        # Define frequency vector
        self.freqs = (xp.arange(self.num) + self.k_min)*self.df

        '''
          Add noise to the data and whiten frequency waveform with theoretical PSD.
        '''
        #noise = AnalyticNoise(self.freqs, 'MRDv1')
        #psd_A, psd_E = noise.psd(option="A"), noise.psd(option="E")
        noise = AnalyticNoise(self.f0, 'MRDv1')
        psd_A, psd_E = noise.psd(option="A"), noise.psd(option="E")

        batch_size = gb1.A.shape[0]
       
        #noiseA = sample_noise(xp.array(psd_A), self.df, self.dt, )
        #noiseE = sample_noise(xp.array(psd_E), self.df, self.dt, )
 
        noiseA = sample_noise(psd_A, self.df, self.dt, batch_size, self.num)
        noiseE = sample_noise(psd_E, self.df, self.dt, batch_size, self.num)
        
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

            #xA = (xA + noiseA)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_A))  # check if we need factor of dt here of not
            #xE = (xE + noiseE)*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(psd_E))  # check if we need factor of dt here of not
            xA = (xA + noiseA[i,:])*xp.sqrt(4.0*self.df)/xp.sqrt(psd_A)  # check if we need factor of dt here of not
            xE = (xE + noiseE[i,:])*xp.sqrt(4.0*self.df)/xp.sqrt(psd_E)  # check if we need factor of dt here of not

            A_white[i,:] = xA        
            E_white[i,:] = xE
  
#        if iteration == 0:
#            if self.config['training']['resume'] == 0:
#                self.Ar_mean = xp.mean(xp.real(A_white), axis=0)
#                self.Er_mean = xp.mean(xp.real(E_white), axis=0)
#                self.Ar_std  = xp.std(xp.real(A_white), axis=0)
#                self.Er_std  = xp.std(xp.real(E_white), axis=0)     
#                self.Aim_mean = xp.mean(xp.imag(A_white), axis=0)
#                self.Eim_mean = xp.mean(xp.imag(E_white), axis=0)
#                self.Aim_std  = xp.std(xp.imag(A_white), axis=0)
#                self.Eim_std  = xp.std(xp.imag(E_white), axis=0)     
#
#                np.savetxt('means_wf' + self.config['saving']['label'] + '.txt', np.c_[self.Ar_mean.get(), self.Er_mean.get(), self.Aim_mean.get(), self.Eim_mean.get()])
#                np.savetxt('stds_wf' + self.config['saving']['label'] + '.txt', np.c_[self.Ar_std.get(), self.Er_std.get(), self.Aim_std.get(), self.Eim_std.get()])
#
#            else:
#
#                self.Ar_mean, self.Er_mean, self.Aim_mean, self.Eim_mean = xp.loadtxt('means_wf' + self.config['saving']['label'] + '.txt')
#                self.Ar_std, self.Er_std, self.Aim_std, self.Eim_std = xp.loadtxt('stds_wf' + self.config['saving']['label'] + '.txt')
        A_white_log = cp.sign(cp.real(A_white))*cp.log(cp.abs(cp.real(A_white)))
        print('A_white_log.shape = ', A_white_log.shape) 
        plt.figure()
        plt.plot(cp.real(A_white)[0,:].get())
        plt.savefig('A_white.png')
        exit()

        plt.figure()
        plt.plot(A_white_log[0,:].get())
        plt.savefig('A_white_log.png')
        exit() 
 
        return A_white, E_white

        
    def get_params(self):
        return self.param_batch

    def get_param_label(self):
        return self.parameter_labels

    def get_param_mean(self):
        return self.parameters_mean

    def get_param_std(self):
        return self.parameters_std

#    def get_wf_mean(self):
#        return self.Ar_mean, self.Er_mean, self.Aim_mean, self.Eim_mean

#    def get_wf_std(self):
#        return self.Ar_std, self.Er_std, self.Aim_std, self.Eim_std


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
            truths = np.array((f0, fdot, np.sin(beta), lam))
            #truths = np.array([f0, fdot, np.sin(beta), lam, np.cos(iota), amp, phi0, psi])

        else:
            a1, a2, a3, a4 = transform_params(torch.as_tensor(amp).type(self.dtype), \
                                              torch.as_tensor(np.cos(iota)).type(self.dtype), \
                                              torch.as_tensor(phi0).type(self.dtype), \
                                              torch.as_tensor(psi).type(self.dtype))

            truths = np.array([f0, fdot, np.sin(beta), lam, a1.cpu().detach().numpy(), a2.cpu().detach().numpy(), a3.cpu().detach().numpy(), a4.cpu().detach().numpy()])
    
        # Create waveform
        gb = GBGPU(use_gpu=True)
        gb.run_wave(*params, N = self.num_f, dt = self.dt, T = self.Tobs, oversample = 2)#oversample=2)

        A_out = xp.zeros((1, self.num), dtype=xp.complex128)
        E_out = xp.zeros((1, self.num), dtype=xp.complex128)

        i_start = (gb.start_inds.get() - self.k_min).astype(np.int32)
        i_end = (gb.start_inds.get() - self.k_min + gb.N).astype(np.int32)

        A_out[0, i_start[0] : i_end[0]] = gb.A
        E_out[0, i_start[0] : i_end[0]] =  gb.E
        
        noise = AnalyticNoise(self.freqs, 'MRDv1')
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
     
        noiseA = sample_noise(noisevals_A, self.df, self.dt, 1, self.num)#, A_out.shape)
        noiseE = sample_noise(noisevals_E, self.df, self.dt, 1, self.num)#, E_out.shape)

        #A_white = (A_out/self.dt + noiseA) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        #E_white = (E_out/self.dt + noiseE) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))

        #A_white = (A_out + noiseA) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        #E_white = (E_out + noiseE) * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))
 
        A_white = (A_out + noiseA) * xp.sqrt(4.0*self.df)/xp.sqrt(noisevals_A)
        E_white = (E_out + noiseE) * xp.sqrt(4.0*self.df)/xp.sqrt(noisevals_E)
 
        # Have to check if this thing and stuff that goes as input to the network is the same
  
        return A_white, E_white, truths
