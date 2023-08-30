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
   #n_real = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(2.0*np.sqrt(df)), size=sample_shape)
   #n_imag = xp.random.normal(loc=0.0, scale=np.sqrt(variance)/(2.0*np.sqrt(df)), size=sample_shape)


   #n_real = xp.random.normal(loc=0.0, scale=np.sqrt(variance), size=sample_shape)
   #n_imag = xp.random.normal(loc=0.0, scale=np.sqrt(variance), size=sample_shape)

   return n_real+1j*n_imag
    

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
            params.append(self.config_data['default']['f0'])

        if self.config_data['estimate']['fdot']:

            param_min.append(self.config_data['limits']['min']['fdot'])
            param_max.append(self.config_data['limits']['max']['fdot'])
            parameter_labels.append('fdot')
            params.append(self.config_data['default']['fdot'])

        if self.config_data['estimate']['beta']:

            param_min.append(0.0)
            param_max.append(1.0)
            #param_min.append(self.config_data['limits']['min']['beta_sin'])
            #param_max.append(self.config_data['limits']['max']['beta_sin']) 
            parameter_labels.append('sbeta')
            params.append(np.sin(self.config_data['default']['beta']))

#           param_min.append(self.config_data['limits']['min']['beta_cos'])
#           param_max.append(self.config_data['limits']['max']['beta_cos']) 
#           #param_min.append(self.config_data['limits']['min']['beta'] * np.pi)
#           #param_max.append(self.config_data['limits']['max']['beta'] * np.pi)
#           parameter_labels.append('cbeta')
#           params.append(np.sin(self.config_data['default']['beta']))


        if self.config_data['estimate']['lam']:

            param_min.append(0.0)
            param_max.append(1.0) 
            #param_min.append(self.config_data['limits']['min']['lam'])
            #param_max.append(self.config_data['limits']['max']['lam'] * np.pi)
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
            # amp_norm_log = torch.from_numpy(normalise(amp_log, self.config_data['limits']['min']['amp'], self.config_data['limits']['max']['amp'])).type(self.dtype).view(-1,1) 
            amp_norm = torch.from_numpy(amp).type(self.dtype).view(-1,1)/(10**self.config_data['limits']['max']['amp'])
            #param_batch = torch.cat([param_batch, amp_norm], dim=1)
        else:
            amp = np.full((N), self.config_data['default']['amp'])

        if self.config_data['estimate']['f0']:
            f0 = np.random.uniform(self.config_data['limits']['min']['f0'], self.config_data['limits']['max']['f0'], N)
            f0_norm = torch.from_numpy(normalise(f0, self.config_data['limits']['min']['f0'], self.config_data['limits']['max']['f0'])).type(self.dtype).view(-1,1)
            #param_batch = torch.cat([param_batch, f0_norm], dim=1) 
        else:
            f0 = np.full((N), self.config_data['default']['f0'])

        if self.config_data['estimate']['fdot']:
            fdot = np.random.uniform(self.config_data['limits']['min']['fdot'], self.config_data['limits']['max']['fdot'], N)
            fdot_norm = torch.from_numpy(normalise(fdot, self.config_data['limits']['min']['fdot'], self.config_data['limits']['max']['fdot'])).type(self.dtype).view(-1,1)
            #param_batch = torch.cat([param_batch, fdot_norm], dim=1) 
        else:
            fdot = np.full((N), self.config_data['default']['fdot'])

        #if self.config_data['estimate']['fddot']:
        #    fddot = np.random.uniform(self.config_data['limits']['min']['fddot'], self.config_data['limits']['max']['fddot'], N)
        #    fddot_norm = torch.from_numpy(normalise(fddot, 10.0**(self.config_data['limits']['min']['fddot']), 10.0**(self.config_data['limits']['max']['fddot']))).type(self.dtype).view(-1,1)
        #    param_batch = torch.cat([param_batch, amp_norm], dim=1)
        #else:
        #    fddot = np.full((N), self.config_data['default']['fddot'])

        if self.config_data['estimate']['beta']:
            beta_sin = np.random.uniform(self.config_data['limits']['min']['beta_sin'], self.config_data['limits']['max']['beta_sin'], N)
            beta = np.arcsin(beta_sin)
            #beta_norm = torch.from_numpy(normalise(beta, self.config_data['limits']['min']['beta'] * np.pi, self.config_data['limits']['max']['beta'] * np.pi)).type(self.dtype).view(-1,1)
            #beta_norm = torch.from_numpy(normalise(beta_sin, self.config_data['limits']['min']['beta_sin'], self.config_data['limits']['max']['beta_sin'])).type(self.dtype).view(-1,1) 
            #param_batch = torch.cat([param_batch, beta_norm], dim=1) 
        else:
            beta = np.full((N), self.config_data['default']['beta'])

#        if self.config_data['estimate']['beta']:
#            beta_cos = np.random.uniform(self.config_data['limits']['min']['beta_cos'], self.config_data['limits']['max']['beta_cos'], N)
#            beta = np.arccos(beta_cos)
#            #beta_norm = torch.from_numpy(normalise(beta, self.config_data['limits']['min']['beta'] * np.pi, self.config_data['limits']['max']['beta'] * np.pi)).type(self.dtype).view(-1,1)

#        else:
#            beta = np.full((N), self.config_data['default']['beta'])

        if self.config_data['estimate']['lam']:
            lam = np.random.uniform(self.config_data['limits']['min']['lam'] * np.pi, self.config_data['limits']['max']['lam'] * np.pi, N)
            #lam_norm = torch.from_numpy(normalise(lam, self.config_data['limits']['min']['lam'] * np.pi, self.config_data['limits']['max']['lam'] * np.pi)).type(self.dtype).view(-1,1)
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

        # Combine the batch that we are going to sample
        #self.param_batch = torch.cat([param_batch, a1, f0_norm, fdot_norm, beta_norm, lam_norm, a2, a3, a4], dim=1)
        self.param_batch = torch.cat([param_batch, a1, f0_norm, fdot_norm, torch.from_numpy(beta).type(self.dtype).view(-1,1), torch.from_numpy(lam).type(self.dtype).view(-1,1), a2, a3, a4], dim=1)


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
        #print('self.num = ', self.num)

        #print('f_start = ', (self.config_data['limits']['min']['f0']/self.df - 258)*self.df)
        #print('f_end = ', (self.config_data['limits']['max']['f0']/self.df + 258)*self.df) 

        self.freqs = (np.arange(self.num) + self.k_min)*self.df
       
        A_out = xp.zeros((batch_size, self.num), dtype=xp.complex128)
        E_out = xp.zeros((batch_size, self.num), dtype=xp.complex128)

        i_start = (gb.start_inds.get() - self.k_min).astype(np.int32)
        i_end = (gb.start_inds.get() - self.k_min + gb.N).astype(np.int32)
       
        #print('gb.A[0].shape = ', gb.A[0].shape)
 
        # THRES HAS TO BE A WAY TO DO IT WITHOUT THE LOOP!!!
        for i in range(batch_size):
            A_out[i, i_start[i] : i_end[i]] = gb.A[i]
            E_out[i, i_start[i] : i_end[i]] =  gb.E[i]

        '''
          Add noise to the data   #Whiten frequency waveform with theoretical PSD.
        ''' 
        # Load estimates of the noise PSDs for the different channels to whiten the data
        #noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        #noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs) #np.array([self.config_data['default']['f0']]))#self.freqs)
        
        noise = AnalyticNoise(self.freqs)
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
     
        

        #A_white = A_out * xp.sqrt(2.0*self.dt)/xp.sqrt(xp.array(noisevals_A))
        #E_white = E_out * xp.sqrt(2.0*self.dt)/xp.sqrt(xp.array(noisevals_E))
 
        noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)#, E_out.shape)

        A_noise = A_out + noiseA
        E_noise = E_out + noiseE

        A_white = A_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        E_white = E_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))


        #for ip in range(3):

        #    print('amp = ', amp[0:3])
        #    print('freq = ', f0[0:3])
        #    plt.figure()
        #    plt.plot(xp.asnumpy(xp.sqrt(xp.real(A_out[ip,:])*xp.real(A_out[ip,:]) + xp.imag(A_out[ip,:])*xp.imag(A_out[ip,:]))))
        #    plt.savefig('A'+str(ip)+'_out.png')
        #    plt.figure()
        #    plt.plot(xp.asnumpy(xp.arctan2(xp.imag(A_out[ip,:]),xp.real(A_out[ip,:]))))
        #    plt.savefig('A'+str(ip)+'_out_phase.png')
        #    plt.figure()
        #    plt.plot(xp.asnumpy(xp.imag(A_out[ip,:])))
        #    plt.savefig('A'+str(ip)+'_out_imag.png')
        #    plt.figure()
        #    plt.plot(xp.asnumpy(xp.real(A_out[ip,:])))
        #   plt.savefig('A'+str(ip)+'_out_real.png')
 
    
        #    plt.figure()
        #    plt.plot(xp.asnumpy(xp.sqrt(xp.real(A_noise[ip,:])*xp.real(A_noise[ip,:]) + xp.imag(A_noise[ip,:])*xp.imag(A_noise[ip,:]))))
        #    plt.savefig('A'+str(ip)+'_noise.png')
    
        #    plt.figure()
        #    plt.plot(xp.asnumpy(xp.sqrt(xp.real(A_white[ip,:])*xp.real(A_white[ip,:]) + xp.imag(A_white[ip,:])*xp.imag(A_white[ip,:]))))
        #    plt.savefig('A'+str(ip)+'_white.png')

  
        
        #plt.figure()
        #plt.loglog(np.abs(A_noise[0,:].get()))
        #plt.savefig('A_noise.png')

   
 #       print('max Amplitude = ', 10**(self.config_data['limits']['max']['amp'])/np.sqrt(2.0*self.df))
        #A_scale = A_noise/(10**(self.config_data['limits']['max']['amp'])/(2.0*np.sqrt(self.df)))
        #E_scale = E_noise/(10**(self.config_data['limits']['max']['amp'])/(2.0*np.sqrt(self.df)))


        #plt.figure()
        #plt.loglog(self.freqs,np.abs(A_out[0,:].get()))
        #plt.savefig('A_out.png')

        #plt.figure()
        #plt.plot(self.freqs,np.abs(A_noise[0,:].get()))
        #plt.savefig('A_noise.png')

        #plt.figure()
        #plt.loglog(np.abs(A_scale[0,:].get()))
        #plt.savefig('A_scale.png')

        #return A_noise, E_noise, A_out, E_out
        return A_white, E_white  #xp.real(A_noise), xp.real(E_noise), xp.imag(A_noise), xp.imag(E_noise)


    def get_params(self):
        return self.param_batch
 

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

        # Load estimates of the noise PSDs for the different channels to whiten the data
        #noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        #noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, np.array([self.config_data['default']['f0']])) # self.freqs)
        
        noise = AnalyticNoise(self.freqs)
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
     
        noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)#, E_out.shape)

        A_noise = A_out + noiseA
        E_noise = E_out + noiseE
  
        A_white = A_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_A))
        E_white = E_noise * xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(xp.array(noisevals_E))
       
        #A_scale = A_out/(10**(self.config_data['limits']['max']['amp'])/(2.0*np.sqrt(self.df)))
        #E_scale = E_out/(10**(self.config_data['limits']['max']['amp'])/(2.0*np.sqrt(self.df)))


        #print('noisevals_A = ', noisevals_A)
        #print('noisevals_E = ', noisevals_E)

        #A_white = A_out * xp.sqrt(2.0*self.dt)/xp.sqrt(xp.array(noisevals_A))
        #E_white = E_out * xp.sqrt(2.0*self.dt)/xp.sqrt(xp.array(noisevals_E))
      
        return A_white, E_white
        #return A_noise, E_noise, A_out, E_out

