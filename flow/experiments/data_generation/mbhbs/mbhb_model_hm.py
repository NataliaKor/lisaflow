''':wq

  Generation of the MBHB waveforms on the fly.
'''

import numpy as np
import cupy as xp
from copy import deepcopy

import matplotlib.pyplot as plt

import torch

#from astropy.cosmology import Planck18  
from astropy.cosmology import FlatLambdaCDM

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import *
from bbhx.utils.transform import *

#import lisabeta.lisa.ldctools as ldctools
#import lisabeta.lisa.lisa as lisa

import time

from flow.utils.noisemodel import *
from flow.utils.datagenutils_mbhb import *
from flow.utils.transform_to_as import *

#import sys
#sys.path.append('..')
from flow.experiments.data_generation.base import Source

#np.random.seed(111222)

class MBHB_gpu(Source):
    def __init__(self, config, config_params, dtype):
         super().__init__()
   
         self.config_params = config_params
         self.config = config
         self.dtype = dtype
 
         self.dt = self.config_params['tvec']['dt']
         self.Tobs = self.config_params['tvec']['Tobs']
         self.df = 1./self.Tobs         

         self.Nfreq = int(self.Tobs / self.dt)
         self.freqs = xp.fft.rfftfreq(self.Nfreq, self.dt)

#         self.t_start = 0.0
#         self.t_end = 1.0
#         self.t_length = 2**12
         #self.t_resolution = 2**15

         self.mbhb = None
      
         #self.freqs = None
         self.params = None # Values of paramteres that are passed to the waveform generator
         self.params_batch = None # Batch with the values of parameters which are passed for training

         self.params_mean = [] # mean values for the batch of parameters
         self.params_std = [] # standard devitation 
         self.params_label = []
         self.params_min = []
         self.params_max = []       

         # Have to record mean value and standard deviation
      
    # Sample from the prior of auxillary parameters
    def sample_from_prior(self, N, iteration):
        '''
          Parameters that are not estimated are marginalised over, therefore we sample them also from the prior
          We will sample here directly from the auxiliary parameters.

        '''
        params = None
        
        # Initilise parameters with default values
        params_default = self.config_params['default']
        params = params_default.copy()
        
        params['dist'] = DL(float(params['z']))[0] * PC_SI * 1e6 # DL Converts to Mpc

        # Convert to Sylvains parameters
        Sylvain = False
        if Sylvain == True:
           params_map = transform_params_mbhb(params, 'forward', 1e-20, self.dtype) 
        
        params_min_all = self.config_params['limits']['min']
        params_max_all = self.config_params['limits']['max'] 

        angles = {'phi_ref', 'phi', 'lam', 'psi'}

        num_params = 0
        for key, value in self.config_params['estimate'].items():
            num_params = num_params + value
        #num_params = self.config['model']['base']['params']

        self.params_batch = np.zeros((N, num_params))

        i = 0
        # Sample parameters the ones we choose to vary
        for key, value in self.config_params['estimate'].items():
            if value == 1 :
                if key in angles:
                    c_pi = np.pi
                else: c_pi = 1.      
                params[key] = np.random.uniform(c_pi*float(params_min_all[key]), c_pi*float(params_max_all[key]), N) 
                # For the first iteration record values for normalisation and labels to be able to restore the values to original range
                if iteration == 0:
                    # Estimate and record mean and std values
                    self.params_mean.append(np.mean(params[key], axis=0))
                    self.params_std.append(np.std(params[key], axis=0))
                    #self.params_min.append(c_pi*params_min_all[key])
                    #self.params_max.append(c_pi*params_max_all[key]) 
                    self.params_label.append(key)
                # Standerdise parameters
                self.params_batch[:,i] = (params[key] - self.params_mean[i]) / self.params_std[i]              
                # Normalise
                #self.params_batch[:,i]  = normalise_par(self.params_batch[:,i], params_min_all[key]*c_pi, params_max_all[key]*c_pi)
                i+=1
            else:
                # If we do not estimate the parameter, we record the default value
                params[key] = np.full((N), params_default[key])

        np.savetxt(self.config['saving']['save_root'] + 'mean_' + self.config['saving']['label'] + '.txt', self.params_mean)
        np.savetxt(self.config['saving']['save_root']+ 'std_' + self.config['saving']['label'] + '.txt', self.params_std)
        #print('self.params_mean = ', self.params_mean)
        #print('self.params_std = ', self.params_std)
        #print('self.params_label = ', self.params_label)
              

        # TODO Choose if we sample from the SSB frame or from the LISA frame
        #sample_in_frame = 'SSB'
        #if sample_in_frame == 'LISA':
 
        #else:
        #    print('Such frame is not yet defined')

        # Fixed values that we have to sample in the future 
        # NOTE change sampling of the distance from redshift to sampling from co-moving volume`
        # params['dist'] = DL(params['z'])[0] * PC_SI * 1e6 # DL Converts to Mpc 
      
        # NOTE this is not done exactly correct. We have to change t_ref because it will change the response but because the signal is very short 
        # and the prior is very narrow we can assume that the detector is quasi stationary.    
        # Convert parameters to a different frame
        if self.config_params['frame'] == 'LISA':
            params['t_ref'], params['lam'], params['beta'], params['psi'] = LISA_to_SSB(params['t_ref'], params['lam'], np.arcsin(params['beta']), params['psi'], t0=0.)
            #params['t_ref'], params['lam'], params['beta'], params['psi'] = lisa.lisatools.ConvertLframeParamsToSSBframe(params['t_ref'], params['lam'], np.arcsin(params['beta']), params['psi'], constellation_ini_phase=0.) 
        else:
            params['beta'] = np.arcsin(params['beta'])
        params['inc'] = np.arccos(params['inc'])
        #if sample_in_frame == 'SSB':
        #    psi, inc = ldctools.AziPolAngleL2PsiIncl(beta, lam, the, phi)
        #    tcL_var, lamL, betaL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(tc_var, lam, beta, psi, 0.0)  # Check what exactly is zero #value
        #    inc_cos = np.cos(inc)
        #    betaL_sin = np.sin(betaL)
        #params['psi'], params['inc'] = ldctools.AziPolAngleL2PsiIncl(np.arcsin(params['beta']), params['lam'], np.arccos(params['the']), params['phi'])

        try:
            self.params = params
            params = None
        except ValueError: print('Parameters not defined')


    def freqwave_AE(self):


        #self.freqs = xp.arange(1e-5, 0.05, self.df)

        #Nfreq = int(self.Tobs / self.dt)
        #self.freqs = xp.fft.rfftfreq(Nfreq, self.dt)

        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True)

        modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

        # Convert mu and q to m1 and m2
        m1 = self.params['mu'] * ((self.params['q'] + 1)**0.2)/self.params['q']**0.6
        m2 = self.params['mu'] * self.params['q']**0.4 * (self.params['q'] + 1)**0.2

        self.mbhb = wave_gen(m1, m2, self.params['a1'], self.params['a2'], self.params['dist'],
                                      self.params['phi_ref'], self.params['f_ref'], self.params['inc'], self.params['lam'],
                                      self.params['beta'], self.params['psi'], self.params['t_ref'], freqs = self.freqs,
                                      modes = modes, direct=False, fill=True, length=1024)#[0] 

        #sampling_parameters = xp.vstack((mu, q, a1, a2, inc_cos, lamL, betaL_sin, psiL, phi_ref, offset)).T
 
        # Return the set of standardised parameters to sample
        #self.param_batch = (sampling_parameters - self.parameters_mean) / self.parameters_std


    # Create waveform combinations
    def timewave_AE(self):
   
        self.df = self.freqs[2] - self.freqs[1]

        noise = AnalyticNoise(self.freqs[1:], 'MRDv1')
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")

        # TODO implement my own noise on the GPU !!!
        noiseA_cp = xp.asarray(noisevals_A)
        noiseE_cp = xp.asarray(noisevals_E)

        # Shifting by tc and not tc_var
        t_shift = xp.asarray(self.params['t_ref']) + 5000.0 + 2.0*self.dt*self.Nfreq #  + 5000.0 # -2.0*self.dt*self.t_resolution

        #shift = xp.exp(1j*2.0*np.pi*self.freqs*t_shift) 
        wf_shift = xp.exp(1j*2.0*np.pi*xp.matmul(t_shift.reshape(-1,1), self.freqs.reshape(1,-1)))


        # Whiten the waveforms
        # CHECK WHAT WILL HAPPEN IF I REMOVE dt FACTOR
        Afs_white = self.mbhb[:,0,1:]*xp.sqrt(4.0*self.df)/xp.sqrt(noiseA_cp) # * self.dt
        Efs_white = self.mbhb[:,1,1:]*xp.sqrt(4.0*self.df)/xp.sqrt(noiseE_cp) # * self.dt 
    
        #plt.loglog(freq_new, np.abs(wave_all), '--', label="BBHx")
        #plt.xlabel("f [Hz]", fontsize=14)
        #plt.ylabel(r"$\tilde{h}(f)$ (Hz$^{-1/2}$) A-TDI", fontsize=14)
        #plt.legend()
        #plt.sacefig('')
 
        #Ats_arr = xp.fft.irfft(xp.c_[xp.zeros(Afs_white.shape[0]), Afs_white]*self.wf_shift, axis = 1)
        #Ets_arr = xp.fft.irfft(xp.c_[xp.zeros(Efs_white.shape[0]) ,Efs_white]*self.wf_shift, axis = 1)
        Ats_arr = xp.fft.irfft(xp.c_[xp.zeros(Afs_white.shape[0]), Afs_white]*wf_shift, axis = 1)
        Ets_arr = xp.fft.irfft(xp.c_[xp.zeros(Efs_white.shape[0]), Efs_white]*wf_shift, axis = 1)
 
        # Shift time domain waveform such that the merger is not at the end of the waveform 
        #ts_mbhb = xp.c_[Ats_arr[:,-10000], Ats_arr[:,:1000], Ets_arr[:,-10000], Ets_arr[:,:1000]]
        ts_mbhb = xp.c_[Ats_arr, Ets_arr]

        #print('ts_mbhb.shape = ', ts_mbhb.shape)
        #plt.figure()
        #plt.plot(ts_mbhb[0,:].get())
        #plt.savefig('ts_mbhb0.png')
        #plt.close()

        #plt.figure()
        #plt.plot(ts_mbhb[1,:].get())
        #plt.savefig('ts_mbhb1.png')
        #plt.close()

        #plt.figure()
        #plt.plot(ts_mbhb[2,:].get())
        #plt.savefig('ts_mbhb2.png')
        #plt.close()

        #plt.figure()
        #plt.plot(ts_mbhb[3,:].get())
        #plt.savefig('ts_mbhb3.png')
        #plt.close()

        #plt.figure()
        #plt.plot(ts_mbhb[4,:].get())
        #plt.savefig('ts_mbhb4.png')
        #plt.plot(ts_mbhb[5,:].get())
        #plt.savefig('ts_mbhb0_5.png')
        #plt.close()
    

        #t1 = time.time()
        return ts_mbhb

    def get_params(self):
        return self.params_batch

    def get_param_label(self):
        return self.params_label

    #def get_param_min(self):
    #    return self.params_min

    #def get_param_max(self):
    #    return self.params_max

    def get_param_mean(self):
        return self.params_mean

    def get_param_std(self):
        return self.params_std

    def get_freqs(self):
        return self.freqs
    

