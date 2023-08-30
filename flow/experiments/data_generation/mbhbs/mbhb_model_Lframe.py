'''
  Generation of the MBHB waveforms on the fly.
'''

import numpy as np
import cupy as xp

import matplotlib.pyplot as plt

import torch

#from astropy.cosmology import Planck18  
from astropy.cosmology import FlatLambdaCDM

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import *
from bbhx.utils.transform import *

import lisabeta.lisa.ldctools as ldctools
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
import lisabeta.lisa.lisa as lisa
import lisabeta.tools.pytools as pytools

import time

from flow.utils.noisemodel import *

#import sys
#sys.path.append('..')
from data_generation.base import Source

#np.random.seed(111222)


# Convert red shift to distance
def DL(z):

    ldc_cosmo = FlatLambdaCDM(H0=67.1, Om0=0.3175)
    quantity = ldc_cosmo.luminosity_distance(z)
    return quantity.value, quantity.unit



# Create samples of the noise with the defined variance
def sample_noise(psd, df):

   n_real = xp.random.normal(loc=0.0, scale=np.sqrt(psd)/np.sqrt(2.0*df))
   n_imag = xp.random.normal(loc=0.0, scale=np.sqrt(psd)/np.sqrt(2.0*df))

   return n_real+1j*n_imag


# Normalise parameters to be from 0 to


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
         self.mbhb_orig = None
         self.freqs = None
         self.param_batch = None
         self.offset = 0
         self.t_ref = 0
         self.df = 0.0

    # TODO replace it as a loop, for that I have to be able to import constants in the config file
    # Return parameter ranges and labels for plotting and rescating
    def param_ranges(self):
  
        # Define values for fixed parameters
        # and boundaries for varied parameters
        parameter_labels = []
        param_min = []
        param_max = []
        params = []

         
        if self.config_data['estimate']['mu']:
 
            param_min.append(self.config_data['limits']['min']['mu'])
            param_max.append(self.config_data['limits']['max']['mu'])
            parameter_labels.append('mu')
            params.append(self.config_data['default']['mu'])
 
        if self.config_data['estimate']['q']:
 
            param_min.append(self.config_data['limits']['min']['q'])
            param_max.append(self.config_data['limits']['max']['q'])
            parameter_labels.append('q')
            params.append(self.config_data['default']['q'])
 
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

        if self.config_data['estimate']['z']:
 
            param_min.append(self.config_data['limits']['min']['z'])
            param_max.append(self.config_data['limits']['max']['z'])
            parameter_labels.append('z')
            params.append(self.config_data['default']['z']) 


        if self.config_data['estimate']['phi_ref']:
 
            param_min.append(self.config_data['limits']['min']['phi_ref'] )
            param_max.append(self.config_data['limits']['max']['phi_ref'] * np.pi)
            parameter_labels.append('phi_ref')
            params.append(self.config_data['default']['phi_ref'])
 

        # TODO make a choise which refrence frame to choose
        # Be careful that here me sample in the LISA reference frame
        if self.config_data['estimate']['betaL']:
 
            param_min.append(self.config_data['limits']['min']['betaL_sin'])
            param_max.append(self.config_data['limits']['max']['betaL_sin'])
            parameter_labels.append('sbetaL')
            params.append(np.sin(self.config_data['default']['betaL'])) 


        # TODO make a choise which refrence frame to choose
        # Be careful that here me sample in the LISA reference frame 
        if self.config_data['estimate']['lamL']:
 
            param_min.append(self.config_data['limits']['min']['lamL'] * np.pi)
            param_max.append(self.config_data['limits']['max']['lamL'] * np.pi)
            parameter_labels.append('lamL')
            params.append(self.config_data['default']['lamL']) 

        if self.config_data['estimate']['inc']:

            param_min.append(self.config_data['limits']['min']['inc_cos'])
            param_max.append(self.config_data['limits']['max']['inc_cos'])
            parameter_labels.append('cinc')
            params.append(np.cos(self.config_data['default']['inc'])) 

        if self.config_data['estimate']['psiL']:

            param_min.append(self.config_data['limits']['min']['psiL'])
            param_max.append(self.config_data['limits']['max']['psiL'] * np.pi)
            parameter_labels.append('psiL')
            params.append(self.config_data['default']['psiL']) 

#        if self.config_data['estimate']['the']:
 
#            param_min.append(self.config_data['limits']['min']['the'])
#            param_max.append(self.config_data['limits']['max']['the'])
#            parameter_labels.append('the')
#            params.append(self.config_data['default']['the']) 

#        if self.config_data['estimate']['phi']:
 
#            param_min.append(self.config_data['limits']['min']['phi'])
#            param_max.append(self.config_data['limits']['max']['phi'])
#            parameter_labels.append('phi')   
#            params.append(self.config_data['default']['phi']) 

#        if self.config_data['estimate']['t_ref']:

#            param_max.append(self.config_data['limits']['max']['t_ref'])
#            parameter_labels.append('t_ref')
#            params.append(self.config_data['default']['t_ref']) 

        if self.config_data['estimate']['t_ref']:
 
            param_min.append(self.config_data['limits']['min']['tL_ref'])
            param_max.append(self.config_data['limits']['max']['tL_ref'])
            parameter_labels.append('tL_ref')
            params.append(self.config_data['default']['tL_ref']) 


        if self.config_data['estimate']['offset']:

            param_min.append(self.config_data['limits']['min']['offset'])
            param_max.append(self.config_data['limits']['max']['offset'])
            parameter_labels.append('offset')
            params.append(self.config_data['default']['offset'])

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
              
        if self.config_data['estimate']['mu']:
            mu = np.random.uniform(self.config_data['limits']['min']['mu'], self.config_data['limits']['max']['mu'], N)
            mu_norm = torch.from_numpy(normalise(mu, self.config_data['limits']['min']['mu'], self.config_data['limits']['max']['mu'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, mu_norm], dim=1)  
        else:    
            mu = np.full((N), self.config_data['default']['mu'])

        if self.config_data['estimate']['q']:
            q = np.random.uniform(self.config_data['limits']['min']['q'], self.config_data['limits']['max']['q'], N)
            q_norm = torch.from_numpy(normalise(q, self.config_data['limits']['min']['q'], self.config_data['limits']['max']['q'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, q_norm], dim=1)  
        else:    
            q = np.full((N), self.config_data['default']['q'])

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
                
        #if self.config_data['estimate']['dist']: 
        #    dist = np.random.uniform(self.config_data['limits']['min']['dist']*PC_SI*1e9, self.config_data['limits']['max']['dist']*PC_SI*1e9, N)
        #    dist_norm = torch.from_numpy(normalise(dist, self.config_data['limits']['min']['dist']*PC_SI*1e9, self.config_data['limits']['max']['dist']*PC_SI*1e9)).type(self.dtype).view(-1,1)
        #    param_batch = torch.cat([param_batch, dist_norm], dim=1)  
        #else:
        #    dist = np.full((N), self.config_data['default']['dist']) * PC_SI * 1e9  

        # Sample in the redshift and then convert to distance
        if self.config_data['estimate']['z']:
            z = np.random.uniform(self.config_data['limits']['min']['z'], self.config_data['limits']['max']['z'], N)
            z_norm = torch.from_numpy(normalise(z, self.config_data['limits']['min']['z'], self.config_data['limits']['max']['z'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, z_norm], dim=1)
            dist = DL(z)[0] * PC_SI * 1e6
        else:
            z = np.full((N), self.config_data['default']['z']) 
            dist = DL(z)[0] * PC_SI * 1e6

        if self.config_data['estimate']['phi_ref']: 
            phi_ref = np.random.uniform(self.config_data['limits']['min']['phi_ref'] , self.config_data['limits']['max']['phi_ref'] * np.pi, N)
            phi_ref_norm = torch.from_numpy(normalise(phi_ref, self.config_data['limits']['min']['phi_ref'], self.config_data['limits']['max']['phi_ref'] * np.pi)).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, phi_ref_norm], dim=1) 
        else:
            phi_ref= np.full((N), self.config_data['default']['phi_ref'])    
                  
        if self.config_data['estimate']['betaL']: 
            betaL_sin = np.random.uniform(self.config_data['limits']['min']['betaL_sin'], self.config_data['limits']['max']['betaL_sin'], N) 
            betL = np.arcsin(betaL_sin)
            betaL_norm = torch.from_numpy(normalise(betaL_sin, self.config_data['limits']['min']['betaL_sin'], self.config_data['limits']['max']['betaL_sin'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, betaL_norm], dim=1)  
        else:
            betL = np.full((N), self.config_data['default']['betaL'])  
          
        if self.config_data['estimate']['lamL']: 
            lamL = np.random.uniform(self.config_data['limits']['min']['lamL'] * np.pi, self.config_data['limits']['max']['lamL'] * np.pi, N)
            lamL_norm = torch.from_numpy(normalise(lamL, self.config_data['limits']['min']['lamL'] * np.pi, self.config_data['limits']['max']['lamL']*np.pi)).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, lamL_norm], dim=1)  
        else:
            lamL = np.full((N), self.config_data['default']['lamL']) 

        if self.config_data['estimate']['inc']:
            inc_cos = np.random.uniform(self.config_data['limits']['min']['inc_cos'], self.config_data['limits']['max']['inc_cos'], N)
            inc = np.arccos(inc_cos)
            inc_norm = torch.from_numpy(normalise(inc_cos, self.config_data['limits']['min']['inc_cos'], self.config_data['limits']['max']['inc_cos'] )).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, inc_norm], dim=1)
        else:
            inc = np.full((N), self.config_data['default']['inc']) 
               
        if self.config_data['estimate']['psiL']:
            psiL= np.random.uniform(self.config_data['limits']['min']['psiL'], self.config_data['limits']['max']['psiL'] * np.pi, N)
            psiL_norm = torch.from_numpy(normalise(psiL, self.config_data['limits']['min']['psiL'], self.config_data['limits']['max']['psiL'] * np.pi)).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, psiL_norm], dim=1)
        else:
            psiL = np.full((N), self.config_data['default']['psiL'])
            
        if self.config_data['estimate']['t_ref']: 
            t_ref = np.random.uniform(self.config_data['limits']['min']['t_ref'], self.config_data['limits']['max']['t_ref'], N)
            t_ref_norm = torch.from_numpy(normalise(t_ref, self.config_data['limits']['min']['t_ref'], self.config_data['limits']['max']['t_ref'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, t_ref_norm], dim=1)  
        else:
            t_ref= np.full((N), self.config_data['default']['t_ref'])                   
            #t_Lframe = lisa.lisatools.tLfromSSBframe(t_ref, lam, bet, constellation_ini_phase=0., frozenLISA=False, tfrozenLISA=None)
         
        if self.config_data['estimate']['tL_ref']: 
            tL_ref = np.random.uniform(self.config_data['limits']['min']['tL_ref'], self.config_data['limits']['max']['tL_ref'], N)
            tL_ref_norm = torch.from_numpy(normalise(tL_ref, self.config_data['limits']['min']['tL_ref'], self.config_data['limits']['max']['tL_ref'])).type(self.dtype).view(-1,1)
            # Convert to SBB reference frame
            #t_ref = lisa.lisatools.tSSBfromLframe(tL_ref, lam, bet, constellation_ini_phase=0.)
            param_batch = torch.cat([param_batch, tL_ref_norm], dim=1)  
        else:
            tL_ref = np.full((N), self.config_data['default']['tL_ref'])                   
            #t_ref = lisa.lisatools.tSSBfromLframe(tL_ref, lam, bet, constellation_ini_phase=0.)
        

        if self.config_data['estimate']['offset']: 
            offset = np.random.uniform(self.config_data['limits']['min']['offset'], self.config_data['limits']['max']['offset'], N)
            offset_norm = torch.from_numpy(normalise(offset, self.config_data['limits']['min']['offset'], self.config_data['limits']['max']['offset'])).type(self.dtype).view(-1,1)
            param_batch = torch.cat([param_batch, offset_norm], dim=1)  
        else:
            offset = np.full((N), self.config_data['default']['offset'])                   
       
        # Calculate chirp mass and mass ratio
        #mu = (m1*m2)**0.6/(m1 + m2)**0.2
        #q = m2/m1 
      

        # log frequencies to interpolate to
        #freq_new = xp.logspace(-4, 0, 10000)
 
        # modes
        # modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        modes = [(2,2)]
      
        t_length = 110000.0  
        #t_length = 36000.0
        n_short  = int(t_length / self.dt)
        freqs_short = xp.fft.rfftfreq(n_short, self.dt)[1:]       
 
        #n = int(t_ref[0] / self.dt)        
        #data_freqs = xp.fft.rfftfreq(n, self.dt)[1:]  # all frequencies except DC

        #t_obs_start = t_length/YRSID_SI

        # Convert chirp mass and mass ratio to m1 and m2
        m1 = mu * ((q + 1)**0.2)/q**0.6
        m2 = mu * q**0.4 * (q + 1)**0.2  


        # Test if lambda angle is in the correct range
        #lam_check = pytools.mod2pi(lam)
        #psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, the, phi)
        #psi_check = pytools.modpi(psi)
        #tL_ref, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(t_ref, lam_check, bet, psi_check, 0.0)
      
        # Calculate true values of beta and lambda in the LISA reference frame
        #psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, the, phi)
        #tc = t_ref 

        #wave_orig = wave_gen(m1, m2, a1, a2,
        #                dist, phi_ref, f_ref, inc, lam, 
        #                bet, psi, tc, freqs = freqs_short,
        #                modes=modes, direct=False, fill=True, squeeze=True, length=1024) #, t_obs_start=t_obs_start) # [0] 

 
        # Calculate the offsets for cuting the data around the time of te coalescence
        #tL_ref0 = np.full((N), self.config_data['default']['tL_ref'])
        #t_ref0 = lisa.lisatools.tSSBfromLframe(tL_ref0, lam, bet, constellation_ini_phase=0.) 
        #self.offset = ((t_ref - t_ref0)/self.dt).astype(int)   
        #t_obs_start = (27*3600 - (t_ref - t_ref0))/YRSID_SI

        # Calculate true values of beta and lambda in the LISA reference frame
        #psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, the, phi)
        #tL_ref, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(t_ref, lam, bet, psi, 0.0)
       

        # NOTE this is not done exactly correct. We have to change t_ref because it will change the response but because the signal is very short 
        # and the prior is very narrow we can assume that the detector is quasi stationary.         
        tcL  = tL_ref # - offset
        #tcL = tL_ref

        #tcL, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(tc, lam, bet, psi, 0.0)  
        tc, lam, bet, psi = lisa.lisatools.ConvertLframeParamsToSSBframe(tcL, lamL, betL, psiL, constellation_ini_phase=0.)
       

        # NOTE not sure if it is correct to put the same t_ref here in each waveform generation
        wave = wave_gen(m1, m2, a1, a2,
                        dist, phi_ref, f_ref, inc, lam, 
                        bet, psi, tc, freqs = freqs_short,
                        modes=modes, direct=False, fill=True, squeeze=True, length=1024) #, t_obs_start=t_obs_start) # [0] 

     

     
        self.mbhb = wave
        #self.freqs = data_freqs
        self.freqs = freqs_short
        self.df = self.freqs[2] - self.freqs[1]
        self.param_batch = param_batch      
        self.t_ref = t_ref 
        self.offset = offset


    def get_params(self):
        return self.param_batch


    def get_AE_freq(self):
        return  self.mbhb[:,0,:], self.mbhb[:,1,:]


    def get_noise_psd(self):
        # Load estimates of the noise PSDs for the different channels to whiten the data
        noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs.get())

        return self.freqs, xp.asarray(noisevals_A), xp.asarray(noisevals_E)
        
    def timewave_AET(self):
        '''
          Whiten frequency waveform with theoretical PSD.
        '''
        #t0 = time.time()
        # Load estimates of the noise PSDs for the different channels to whiten the data
        #noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        #noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs.get())

        noise = AnalyticNoise(self.freqs)
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")

        # TODO implement my own noise on the GPU !!!
        noiseA_cp = xp.asarray(noisevals_A)
        noiseE_cp = xp.asarray(noisevals_E)

        # Whiten the waveforms
        Afs_white = self.mbhb[:,0,:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseA_cp)
        Efs_white = self.mbhb[:,1,:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseE_cp)
       
        # Time shift
        #shift = xp.exp(-1j*2.0*np.pi*xp.matmul(xp.asarray(self.offset).reshape(-1,1),self.freqs.reshape(1,-1)))
       
        # Inverse transform
        #Ats_arr = xp.fft.irfft(Afs_white*shift, axis = 1)
        #Ets_arr = xp.fft.irfft(Efs_white*shift, axis = 1)

        Ats_arr = xp.fft.irfft(Afs_white, axis = 1)
        Ets_arr = xp.fft.irfft(Efs_white, axis = 1)
  
        # Time shift because the waveform was cut
        #t_length = 110000.0
        #t_shift = xp.asarray(self.t_ref) - t_length
        #shift = xp.exp(1j*2.0*np.pi*self.freqs*t_shift) 
        #Ats_arr = xp.fft.irfft(Afs_white*shift, axis = 1)
        
        # Shift time domain waveform such that the merger is not at the end of the waveform 
        #ts_mbhb = xp.c_[Ats_arr[:,-10000], Ats_arr[:,:1000], Ets_arr[:,-10000], Ets_arr[:,:1000]]
        ts_mbhb = xp.c_[Ats_arr, Ets_arr]
       
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
        z = self.config_data['default']['z'] 
        #bet = self.config_data['default']['beta']
        #lam = self.config_data['default']['lam']
        #the = self.config_data['default']['the']
        #phi = self.config_data['default']['phi']
        betL = self.config_data['default']['betaL']
        lamL = self.config_data['default']['lamL'] 
        psiL = self.config_data['default']['psiL']
        inc = self.config_data['default']['inc']

        tL_ref = self.config_data['default']['tL_ref']
        offset = self.config_data['default']['offset']

        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True)
        
        # Convert angles
        #psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, the, phi)
        #tcL, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(t_ref, lam, bet, psi, 0.0)  # Check what exactly is zero value

        tcL = tL_ref
        tc, lam, bet, psi = lisa.lisatools.ConvertLframeParamsToSSBframe(tcL, lamL, betL, psiL, constellation_ini_phase=0.)
        dist =  DL(z)[0] * PC_SI * 1e6

        # log frequencies to interpolate to
        #freq_new = xp.logspace(-4, 0, 10000)
        t_length = 110000.0
        #t_length = 36000.0
        n_short  = int(t_length / self.dt)
        freqs_short = xp.fft.rfftfreq(n_short, self.dt)[1:]

        modes = [(2,2)]

        #n = int(t_ref / self.dt)

        #data_freqs = xp.fft.rfftfreq(n, self.dt)[1:]  # all frequencies except DC
   
        wave = wave_gen(m1, m2, a1, a2,
                        dist, phi_ref, f_ref, inc, lam,
                        bet, psi, tc, freqs = freqs_short,
                        modes=modes, direct=False, fill=True, squeeze=True, length=1024) # [0] 
        print('wave.shape = ', wave.shape)
        # Load estimates of the noise PSDs for the different channels to whiten the data
        #noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        #noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs.get())

        noise = AnalyticNoise(self.freqs)
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")

        # TODO implement my own noise on the GPU !!!
        noiseA_cp = xp.asarray(noisevals_A)
        noiseE_cp = xp.asarray(noisevals_E)

        # Whiten the waveforms
        Afs_white = wave[:,0,:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseA_cp)
        Efs_white = wave[:,1,:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseE_cp)

        #shift = np.exp(1j*2.0*np.pi*data_freqs*offset) 

        # Inverse transform
        #Ats_arr = xp.fft.irfft(Afs_white*shift, axis = 1)
        #Ets_arr = xp.fft.irfft(Efs_white*shift, axis = 1)

        Ats_arr = xp.fft.irfft(Afs_white, axis = 1)
        Ets_arr = xp.fft.irfft(Efs_white, axis = 1)

        # Shift time domain waveform such that the merger is not at the end of the waveform 
        ts_mbhb = xp.c_[Ats_arr, Ets_arr]
      
        return ts_mbhb


#    x_wf = torch.from_numpy(chirplet_normalised(t, Q, t0, f0, fdot)).type(dtype)
#    coeff = torch.matmul(x_wf, torch.from_numpy((1.0/np.sqrt(Sts_reduce))*Vts_reduce).type(dtype)).view(1,-1)

  
