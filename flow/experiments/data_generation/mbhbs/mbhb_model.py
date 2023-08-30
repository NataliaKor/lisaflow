'''
  Generation of the MBHB waveforms on the fly.
'''

import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
import torch

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import *
from bbhx.utils.transform import *

import lisabeta.lisa.ldctools as ldctools
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
import lisabeta.lisa.lisa as lisa

import time

#import sys
#sys.path.append('..')
from data_generation.base import Source

#np.random.seed(111222)

# Normalise parameters to be from 0 to 1
def normalise(par, par_min, par_max):

    return (par - par_min)/(par_max - par_min)


# Will it be better to create the class that stores the values of the saved parameters
# and overwrites only the ones which have to vary.

class MBHB_gpu(Source):
    def __init__(self, config, config_data, dtype):
         super().__init__()
   
         self.dt = config_data['tvec']['dt']
         self.config_data = config_data
         self.dtype = dtype

         self.mbhb = None
         self.freqs = None
         self.param_batch = None

    # TODO replace it as a loop, for that I have to be able to import constants in the config file
    # Return parameter ranges and labels for plotting and rescating
    def param_ranges(self):
  
        # Define values for fixed parameters
        # and boundaries for varied parameters
        parameter_labels = []
        param_min = []
        param_max = []
        params = []

        if self.config_data['estimate']['phi_ref']:
 
            param_min.append(self.config_data['limits']['min']['phi_ref'])
            param_max.append(self.config_data['limits']['max']['phi_ref'])
            parameter_labels.append('phi_ref')
            params.append(self.config_data['default']['phi_ref'])
           
        if self.config_data['estimate']['m1']:
 
            param_min.append(self.config_data['limits']['min']['m1'])
            param_max.append(self.config_data['limits']['max']['m1'])
            parameter_labels.append('m1')
            params.append(self.config_data['default']['m1'])
            
        if self.config_data['estimate']['m2']:
 
            param_min.append(self.config_data['limits']['min']['m2'])
            param_max.append(self.config_data['limits']['max']['m2'])
            parameter_labels.append('m2')
            params.append(self.config_data['default']['m2'])
 
        if self.config_data['estimate']['a1']:
 
            param_min.append(self.config_data['limits']['min']['a1'])
            param_max.append(self.config_data['limits']['max']['a1'])
            parameter_labels.append('a1')
            params.append(self.config_data['default']['a1']) 

        if self.config_data['estimate']['a2']:
 
            param_min.append(self.config_data['limits']['min']['a2'])
            param_max.append(self.config_data['limits']['max']['a2'])
            parameter_labels.append('a2')
            params.append(self.config_data['default']['a2']) 

        if self.config_data['estimate']['dist']:
 
            param_min.append(self.config_data['limits']['min']['dist'])
            param_max.append(self.config_data['limits']['max']['dist'])
            parameter_labels.append('dist')
            params.append(self.config_data['default']['dist']) 

        if self.config_data['estimate']['beta']:
 
            param_min.append(self.config_data['limits']['min']['beta'])
            param_max.append(self.config_data['limits']['max']['beta'])
            parameter_labels.append('beta')
            params.append(self.config_data['default']['beta']) 

        if self.config_data['estimate']['lam']:
 
            param_min.append(self.config_data['limits']['min']['lam'])
            param_max.append(self.config_data['limits']['max']['lam'])
            parameter_labels.append('lam')
            params.append(self.config_data['default']['lam']) 

        if self.config_data['estimate']['theL']:
 
            param_min.append(self.config_data['limits']['min']['theL'])
            param_max.append(self.config_data['limits']['max']['theL'])
            parameter_labels.append('theL')
            params.append(self.config_data['default']['theL']) 

        if self.config_data['estimate']['phiL']:
 
            param_min.append(self.config_data['limits']['min']['phiL'])
            param_max.append(self.config_data['limits']['max']['phiL'])
            parameter_labels.append('phiL')   
            params.append(self.config_data['default']['phiL']) 

        if self.config_data['estimate']['t_ref']:
 
            param_min.append(self.config_data['limits']['min']['t_ref'])
            param_max.append(self.config_data['limits']['max']['t_ref'])
            parameter_labels.append('t_ref')
            params.append(self.config_data['default']['t_ref']) 


        return param_min, param_max, parameter_labels, params
 

 
    def freqwave_AET(self, N):
      
        # Make a choise for the code to work on GPU
        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True)
        param_batch = torch.Tensor().type(self.dtype)

        # Set parameters
        # Choose if we vary parameters or take the default
        # If we vary them we directly pass an array of parameters 
        f_ref   = np.full((N), self.config_data['default']['f_ref'])       
 
        # TODO Is it possible to make a loop here without writing this explicitly  
        if self.config_data['estimate']['phi_ref']: 
            phi_ref = np.random.uniform(self.config_data['limits']['min']['phi_ref'], self.config_data['limits']['max']['phi_ref'] * np.pi, N)
            phi_ref_norm = torch.from_numpy(normalise(phi_ref, self.config_data['limits']['min']['phi_ref'], self.config_data['limits']['max']['phi_ref'] * np.pi)).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, phi_ref_norm], dim=1) 
        else:
            phi_ref= np.full((N), self.config_data['default']['phi_ref'])    
               
        if self.config_data['estimate']['m1']:         
            m1 = np.random.uniform(self.config_data['limits']['min']['m1'], self.config_data['limits']['max']['m1'], N)
            m1_norm = torch.from_numpy(normalise(m1, self.config_data['limits']['min']['m1'], self.config_data['limits']['max']['m1'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, m1_norm], dim=1)  
        else:
            m1 = np.full((N), self.config_data['default']['m1'])  
                 
        if self.config_data['estimate']['m2']:
            m2 = np.random.uniform(self.config_data['limits']['min']['m2'], self.config_data['limits']['max']['m2'], N)
            m2_norm = torch.from_numpy(normalise(m2, self.config_data['limits']['min']['m2'], self.config_data['limits']['max']['m2'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, m2_norm], dim=1)  
        else:    
            m2 = np.full((N), self.config_data['default']['m2'])

        if self.config_data['estimate']['a1']: 
            a1 = np.random.uniform(self.config_data['limits']['min']['a1'], self.config_data['limits']['max']['a1'], N)
            a1_norm = torch.from_numpy(normalise(a1, self.config_data['limits']['min']['a1'], self.config_data['limits']['max']['a1'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, a1_norm], dim=1)  
        else:
            a1 = np.full((N), self.config_data['default']['a1'])   
                
        if self.config_data['estimate']['a2']: 
            a2 = np.random.uniform(self.config_data['limits']['min']['a2'], self.config_data['limits']['max']['a2'], N)
            a2_norm = torch.from_numpy(normalise(a2, self.config_data['limits']['min']['a2'], self.config_data['limits']['max']['a2'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, a2_norm], dim=1)  
        else:
            a2 = np.full((N), self.config_data['default']['a2'])   
                
        if self.config_data['estimate']['dist']: 
            dist = np.random.uniform(self.config_data['limits']['min']['dist']*PC_SI*1e9, self.config_data['limits']['max']['dist']*PC_SI*1e9, N)
            dist_norm = torch.from_numpy(normalise(dist, self.config_data['limits']['min']['dist']*PC_SI*1e9, self.config_data['limits']['max']['dist']*PC_SI*1e9)).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, dist_norm], dim=1)  
        else:
            dist= np.full((N), self.config_data['default']['dist']) * PC_SI * 1e9  
                 
        if self.config_data['estimate']['beta']: 
            beta_sin = np.random.uniform(self.config_data['limits']['min']['beta'], self.config_data['limits']['max']['beta'], N) 
            bet = np.arcsin(beta_sin)
            beta_sin_norm = torch.from_numpy(normalise(beta_sin, self.config_data['limits']['min']['beta'], self.config_data['limits']['max']['beta'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, beta_sin_norm], dim=1)  
        else:
            bet = np.full((N), self.config_data['default']['beta'])  
           
        if self.config_data['estimate']['lam']: 
            lam = np.random.uniform(self.config_data['limits']['min']['lam'], self.config_data['limits']['max']['lam'] * np.pi, N)
            lam_norm = torch.from_numpy(normalise(lam, self.config_data['limits']['min']['lam'], self.config_data['limits']['max']['lam']*np.pi)).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, lam_norm], dim=1)  
        else:
            lam = np.full((N), self.config_data['default']['lam']) 
              
        if self.config_data['estimate']['theL']: 
            theL = np.random.uniform(self.config_data['limits']['min']['theL'], self.config_data['limits']['max']['theL'] * np.pi, N)
            theL_norm = torch.from_numpy(normalise(theL, self.config_data['limits']['min']['theL'], self.config_data['limits']['max']['theL'] * np.pi)).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, theL_norm], dim=1)  
        else:
            theL = np.full((N), self.config_data['default']['theL'])  
              
        if self.config_data['estimate']['phiL']: 
            phiL_cos = np.random.uniform(self.config_data['limits']['min']['phiL'], self.config_data['limits']['max']['phiL'], N)
            phiL = np.arccos(phiL_cos)
            phiL_norm = torch.from_numpy(normalise(phiL, self.config_data['limits']['min']['phiL'], self.config_data['limits']['max']['phiL'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, phiL_norm], dim=1)  
        else:
            phiL = np.full((N), self.config_data['default']['phiL'])       
            
        if self.config_data['estimate']['t_ref']: 
            t_ref = np.random.uniform(self.config_data['limits']['min']['t_ref'], self.config_data['limits']['max']['t_ref'], N)
            t_ref_norm = torch.from_numpy(normalise(t_ref, self.config_data['limits']['min']['t_ref'], self.config_data['limits']['max']['t_ref'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, t_ref_norm], dim=1)  
        else:
            t_ref= np.full((N), self.config_data['default']['t_ref'])                   
  
        # Convert angles
        psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, theL, phiL)

        # Convert parameters to LISA frame 
        # TODO this has to be adjusted to sample for the LISA frame parameters and then transformed to the SSB frame
        # Check if it works correctly for the arrays 
        # tcL, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(tc[i], lambd[i], beta[i], psi, 0.0)
        #tcL, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(t_ref, lam, bet, psi, 0.0)  

        # log frequencies to interpolate to
        #freq_new = xp.logspace(-4, 0, 10000)
 
        # modes
        # modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        modes = [(2,2)]
      
        n = int(t_ref[0] / self.dt)
        
        data_freqs = xp.fft.rfftfreq(n, self.dt)[1:]  # all frequencies except DC

        wave = wave_gen(m1, m2, a1, a2,
                        dist, phi_ref, f_ref, inc, lam, 
                        bet, psi, t_ref, freqs = data_freqs,
                        modes=modes, direct=False, fill=True, squeeze=True, length=1024) # [0] 

        self.mbhb = wave
        self.freqs = data_freqs
        self.param_batch = param_batch       

    def get_params(self):
        return self.param_batch


    def timewave_AET(self):
        '''
          Whiten frequency waveform with theoretical PSD.
        '''
        #t0 = time.time()
        # Load estimates of the noise PSDs for the different channels to whiten the data
        noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs.get())

        # TODO implement my own noise on the GPU !!!
        noiseA_cp = xp.asarray(noisevals_A)
        noiseE_cp = xp.asarray(noisevals_E)

        # Whiten the waveforms
        Afs_white = self.mbhb[:,0,:]*xp.sqrt(2.0*self.dt)/xp.sqrt(noiseA_cp)
        Efs_white = self.mbhb[:,1,:]*xp.sqrt(2.0*self.dt)/xp.sqrt(noiseE_cp)

        # Inverse transform
        Ats_arr = xp.fft.irfft(Afs_white, axis = 1)
        Ets_arr = xp.fft.irfft(Efs_white, axis = 1)
         
        # Shift time domain waveform such that the merger is not at the end of the waveform 
        ts_mbhb = xp.c_[Ats_arr[:,-10000:], Ats_arr[:,:1000], Ets_arr[:,-10000:], Ets_arr[:,:1000]]
 
        #t1 = time.time()
        return ts_mbhb

    def true_data(self):
        '''
          "True" data for testing. Dataset with the default parameters.
        '''      
        f_ref   = self.config_data['default']['f_ref']
        phi_ref = self.config_data['default']['phi_ref']
        m1 = self.config_data['default']['m1']
        m2 = self.config_data['default']['m2']
        a1 = self.config_data['default']['a1']
        a2 = self.config_data['default']['a2']
        dist = self.config_data['default']['dist'] * PC_SI * 1e9
        bet = self.config_data['default']['beta']
        lam = self.config_data['default']['lam']
        theL = self.config_data['default']['theL']
        phiL = self.config_data['default']['phiL']
        t_ref = self.config_data['default']['t_ref']

        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True)
        
        # Convert angles
        psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, theL, phiL)
        tcL, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(t_ref, lam, bet, psi, 0.0)  # Check what exactly is zero value

        # log frequencies to interpolate to
        #freq_new = xp.logspace(-4, 0, 10000)

        modes = [(2,2)]

        n = int(t_ref / self.dt)

        data_freqs = xp.fft.rfftfreq(n, self.dt)[1:]  # all frequencies except DC

        wave = wave_gen(m1, m2, a1, a2,
                        dist, phi_ref, f_ref, inc, lam,
                        bet, psi, t_ref, freqs = data_freqs,
                        modes=modes, direct=False, fill=True, squeeze=True, length=1024) # [0] 
        print('wave.shape = ', wave.shape)
        # Load estimates of the noise PSDs for the different channels to whiten the data
        noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs.get())

        # TODO implement my own noise on the GPU !!!
        noiseA_cp = xp.asarray(noisevals_A)
        noiseE_cp = xp.asarray(noisevals_E)

        # Whiten the waveforms
        Afs_white = wave[:,0,:]*xp.sqrt(2.0*self.dt)/xp.sqrt(noiseA_cp)
        Efs_white = wave[:,1,:]*xp.sqrt(2.0*self.dt)/xp.sqrt(noiseE_cp)

   

        # Inverse transform
        Ats_arr = xp.fft.irfft(Afs_white, axis = 1)
        Ets_arr = xp.fft.irfft(Efs_white, axis = 1)

        # Shift time domain waveform such that the merger is not at the end of the waveform 
        ts_mbhb = xp.c_[Ats_arr[:,-10000:], Ats_arr[:,:1000], Ets_arr[:,-10000:], Ets_arr[:,:1000]]
       
        return ts_mbhb


#    x_wf = torch.from_numpy(chirplet_normalised(t, Q, t0, f0, fdot)).type(dtype)
#    coeff = torch.matmul(x_wf, torch.from_numpy((1.0/np.sqrt(Sts_reduce))*Vts_reduce).type(dtype)).view(1,-1)

  
