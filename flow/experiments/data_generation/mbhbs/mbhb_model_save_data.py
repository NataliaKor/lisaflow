''':wq

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
from flow.experiments.data_generation.base import Source

#np.random.seed(111222)


# Convert red shift to distance
def DL(z):

    ldc_cosmo = FlatLambdaCDM(H0=67.1, Om0=0.3175)
    quantity = ldc_cosmo.luminosity_distance(z)
    return quantity.value, quantity.unit



def sample_noise(variance, df, dt):

   n_real = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(dt*xp.sqrt(4.0*df))) 
   n_imag = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(dt*xp.sqrt(4.0*df))) 

   return n_real+1j*n_imag


# Will it be better to create the class that stores the values of the saved parameters
# and overwrites only the ones which have to vary.

class MBHB_gpu(Source):
    def __init__(self, config, config_data, dtype):
         super().__init__()
   
         self.dt = config_data['tvec']['dt']
         self.config_data = config_data
         self.config = config
         self.dtype = dtype

         self.t_length = 1.0 # One year
         #self.t_resolution = 2**15

         self.mbhb = None
         #self.mbhb_nooff = None # temporary
      
         self.freqs = None
         self.param_batch = None

         # Shift waveform in time
         wf_shift = 1

      
    # Sample from the prior of auxillary parameters
    def sample_from_prior(self, N, iteration):
        '''
          Parameters that are not estimated are marginalised over, therefore we sample them also from the prior
          We will sample here directly from the auxiliary parameters.

        '''
        f_ref   = np.full((N), self.config_data['default']['f_ref'])   # Do we have to vary reference frequency?     
 
        mu    = np.random.uniform(self.config_data['limits']['min']['mu'], self.config_data['limits']['max']['mu'], N)
        q     = np.random.uniform(self.config_data['limits']['min']['q'], self.config_data['limits']['max']['q'], N)
        a1    = np.random.uniform(self.config_data['limits']['min']['a1'], self.config_data['limits']['max']['a1'], N)  
        a2    = np.random.uniform(self.config_data['limits']['min']['a2'], self.config_data['limits']['max']['a2'], N)

        phi_ref = np.random.uniform(self.config_data['limits']['min']['phi_ref'], self.config_data['limits']['max']['phi_ref']*np.pi, N)

        # Choose if we sample from the SSB frame or from the LISA frame
        sample_in_frame = 'SSB'
        if sample_in_frame == 'LISA':
 
            betaL_sin = np.random.uniform(self.config_data['limits']['min']['betaL_sin'], self.config_data['limits']['max']['betaL_sin'], N)
            betaL = np.arcsin(betaL_sin)

            lamL = np.random.uniform(self.config_data['limits']['min']['lamL'] * np.pi, self.config_data['limits']['max']['lamL'] * np.pi, N)

            inc_cos = np.random.uniform(self.config_data['limits']['min']['inc_cos'], self.config_data['limits']['max']['inc_cos'], N)
            incL = np.arccos(inc_cos)
 
            psiL = np.random.uniform(self.config_data['limits']['min']['psiL'], self.config_data['limits']['max']['psiL'] * np.pi, N)
 
            tcL  = np.full((N), self.config_data['default']['tL_ref'])
            offset = np.random.uniform(self.config_data['limits']['min']['offset'], self.config_data['limits']['max']['offset'], N)
            tcL_var = tcL + offset


        elif sample_in_frame == 'SSB':

            beta_sin = np.random.uniform(self.config_data['limits']['min']['beta_sin'], self.config_data['limits']['max']['beta_sin'], N)
            beta = np.arcsin(beta_sin)

            lam = np.random.uniform(self.config_data['limits']['min']['lam'] * np.pi, self.config_data['limits']['max']['lam'] * np.pi, N)

            phi_cos = np.random.uniform(self.config_data['limits']['min']['phi'], self.config_data['limits']['max']['phi'], N)
            phi = np.arccos(phi_cos)
        
            the = np.random.uniform(self.config_data['limits']['min']['the'], self.config_data['limits']['max']['the'] * np.pi, N)

            tc  = np.full((N), self.config_data['default']['t_ref'])
            offset = np.random.uniform(self.config_data['limits']['min']['offset'], self.config_data['limits']['max']['offset'], N)
            tc_var = tc + offset

        else:
            print('Such frame is not yet defined')

    
        m1 = mu * ((q + 1)**0.2)/q**0.6
        m2 = mu * q**0.4 * (q + 1)**0.2

        # Fixed values that we have to sample in the future 
        # NOTE change sampling of the distance from redshift to sampling from co-moving volume
        z = np.full((N), self.config_data['default']['z'])
        dist = DL(z)[0] * PC_SI * 1e6 
      
        # NOTE this is not done exactly correct. We have to change t_ref because it will change the response but because the signal is very short 
        # and the prior is very narrow we can assume that the detector is quasi stationary.    
        # Convert parameters to a different frame
        if sample_in_frame == 'LISA':
            tc_var, lam, beta, psi = lisa.lisatools.ConvertLframeParamsToSSBframe(tcL_var, lamL, betaL, psiL, constellation_ini_phase=0.)
 
        if sample_in_frame == 'SSB':
            psi, inc = ldctools.AziPolAngleL2PsiIncl(beta, lam, the, phi)
            tcL_var, lamL, betaL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(tc_var, lam, beta, psi, 0.0)  # Check what exactly is zero value
            inc_cos = np.cos(inc)
            betaL_sin = np.sin(betaL)

        # Make a choise for the code to work on GPU
        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True)

        # modes
        modes = [(2,2)]    
  
        #freqs_short = xp.fft.rfftfreq(self.t_resolution, d = self.dt)
        n = int(self.t_length * YRSID_SI / self.dt)
        self.freqs =  xp.fft.rfftfreq(n, self.dt)#[1:]
     
        # Shifting by tc and not tc_var
        #t_shift = xp.asarray(tc) - 2.0*self.dt*self.t_resolution + 5000.0
        #t_shift = xp.asarray(tc) - 2.*self.dt*self.t_resolution
        #shift = xp.exp(1j*2.0*np.pi*self.freqs*t_shift) 
        #self.wf_shift = xp.exp(1j*2.0*np.pi*xp.matmul(xp.asarray(t_shift).reshape(-1,1),self.freqs.reshape(1,-1)))
      
      
        # NOTE not sure how much error is introduced when we put the same t_ref here in each waveform generation
        # Bacause the signal is short assume the detector to be quasi stationary
        #delta_t = (2.0*self.dt*self.t_length)/YRSID_SI
        wave = wave_gen(m1, m2, a1, a2,
                        dist, phi_ref, f_ref, inc, lam,
                        beta, psi, tc_var, freqs = self.freqs, # NOTE tc!!!!!!
                        modes=modes, direct=False, fill=True, squeeze=True, length=1024) 
                        #modes=modes, direct=False, fill=True, squeeze=True, length=1024, t_obs_start=tc/YRSID_SI - delta_t, t_obs_end=(tc + 5000.)/YRSID_SI, shift_t_limits=True) #t_obs_start = 2.0*self.dt*self.t_length/YRSID_SI) # shift_t_limits=False, t_obs_start=1./365., t_obs_end=0.)


        self.mbhb = wave
       
        sampling_parameters = xp.vstack((mu, q, a1, a2, inc_cos, lamL, betaL_sin, psiL, phi_ref, offset)).T

        if iteration == 0:
            if self.config['training']['resume'] == 0:
                # Standardise all parameters and record the parameters of the standardasion when it is done for the first time.
                self.parameters_mean = np.mean(sampling_parameters, axis=0)
                print('self.parameters_mean = ', self.parameters_mean)
                self.parameters_std = np.std(sampling_parameters, axis=0)
                print('self.parameters_std = ', self.parameters_std)
                self.parameter_labels = ['mu', 'q', 'a1', 'a2', 'inc_cos', 'lamL', 'betaL_sin', 'psiL', 'phi_ref', 'offset']
                # Record values to the file 
                np.savetxt('means.txt', self.parameters_mean.get())
                np.savetxt('stds.txt', self.parameters_std.get())

            else:
                self.parameters_mean = nploadtxt('means.txt')
                self.parameters_std = np.loadtxt('stds.txt')
           
           
        # Return the set of standardised parameters to sample
        self.param_batch = (sampling_parameters - self.parameters_mean) / self.parameters_std


    # Create waveform combinations
    def create_waveform(self):
   
        self.df = self.freqs[2] - self.freqs[1]

        noise = AnalyticNoise(self.freqs[1:])
        noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")

        # TODO implement my own noise on the GPU !!!
        noiseA_cp = xp.asarray(noisevals_A)
        noiseE_cp = xp.asarray(noisevals_E)


        # Whiten the waveforms
        Afs_white = self.mbhb[:,0,1:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseA_cp) # * self.dt
        Efs_white = self.mbhb[:,1,1:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseE_cp) # * self.dt
      
        Ats_arr = xp.fft.irfft(xp.c_[xp.zeros(Afs_white.shape[0]), Afs_white], axis = 1) #*self.wf_shift, axis = 1)
        Ets_arr = xp.fft.irfft(xp.c_[xp.zeros(Efs_white.shape[0]) ,Efs_white], axis = 1) #*self.wf_shift, axis = 1)
 
        # In the case when we time shift the waveform
        #Ats_arr = xp.fft.irfft(Afs_white*self.wf_shift, axis = 1)
        #Ets_arr = xp.fft.irfft(Efs_white*self.wf_shift, axis = 1)         
 
        #Ats_arr_nooff = xp.fft.irfft(Afs_white_nooff*self.wf_shift, axis = 1)
      
        #plt.figure()
        #plt.plot(Ats_arr[0,:].get())
        #plt.savefig('Aoff_shifted.png')
        #plt.figure()
        #plt.plot(Ats_test[0,:].get())
        #plt.savefig('Aoff.png')
        #plt.figure()
        #plt.plot(Ats_test_nooff[0,:].get())
        #plt.savefig('Anooff0.png')

        # Shift time domain waveform such that the merger is not at the end of the waveform 
        #ts_mbhb = xp.c_[Ats_arr[:,-10000:], Ats_arr[:,:1000], Ets_arr[:,-10000:], Ets_arr[:,:1000]]
        print('Ats_arr.shape = ', Ats_arr.shape)
        ind_start = int((-18000.0 - offset)/self.dt)
        ind_end = int((2000.0 - offset)/self.dt)
        ts_mbhb = xp.c_[Ats_arr[:, ind_start:], Ats_arr[:, :ind_end], Ets_arr[:, ind_start:], Ets_arr[:, ind_end:]]
       
        #t1 = time.time()
        return ts_mbhb


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
    
#
#    def get_noise_psd(self):
#        # Load estimates of the noise PSDs for the different channels to whiten the data
#        noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
#        noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs.get())
#
#        return self.freqs, xp.asarray(noisevals_A), xp.asarray(noisevals_E)

    
    def true_data(self):
        '''
          "True" data for testing. Dataset with the default parameters.
        '''      
        f_ref   = self.config_data['default']['f_ref']
        phi_ref = self.config_data['default']['phi_ref']
        q = self.config_data['default']['q']
        mu = self.config_data['default']['mu']
        #m1 = self.config_data['default']['m1']
        #m2 = self.config_data['default']['m2']
        a1 = self.config_data['default']['a1']
        a2 = self.config_data['default']['a2']
        z = self.config_data['default']['z'] 
       
        sample_in_frame = 'SSB'

        if sample_in_frame == 'LISA':
            betaL = self.config_data['default']['betaL']
            lamL = self.config_data['default']['lamL'] 
            psiL = self.config_data['default']['psiL']
            inc = self.config_data['default']['inc']
            tcL  = self.config_data['default']['tL_ref']
        elif sample_in_frame == 'SSB':
            beta = self.config_data['default']['beta']
            lam = self.config_data['default']['lam']
            the = self.config_data['default']['the']
            phi = self.config_data['default']['phi']
            tc  = self.config_data['default']['t_ref']
        else:
            print('No such reference frame')
   
   
        offset = self.config_data['default']['offset']

        # NOTE temp, test the min and max values of the offset
        #offset_min = -500.0
        #offset_max = 500.0
        #tc_min = tc + offset_min 
        #tc_max = tc + offset_max

        m1 = mu * ((q + 1)**0.2)/q**0.6
        m2 = mu * q**0.4 * (q + 1)**0.2

        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True)
        
        # Convert angles
        if sample_in_frame == 'LISA':
            tc, lam, beta, psi = lisa.lisatools.ConvertLframeParamsToSSBframe(tcL, lamL, betaL, psiL, constellation_ini_phase=0.)
        elif sample_in_frame == 'SSB':
            psi, inc = ldctools.AziPolAngleL2PsiIncl(beta, lam, the, phi)
            tcL, lamL, betaL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(tc, lam, beta, psi, 0.0)  # Check what exactly is zero value
        else: 
            print('No such reference frame')


        #psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, the, phi)
        #tcL, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(t_ref, lam, bet, psi, 0.0)  # Check what exactly is zero value

        #tc, lam, bet, psi = lisa.lisatools.ConvertLframeParamsToSSBframe(tcL, lamL, betL, psiL, constellation_ini_phase=0.)
        dist =  DL(z)[0] * PC_SI * 1e6

        # NOTE This is a long waveform to test 
        # log frequencies to interpolate to
        # freqs_test = xp.logspace(-4, 0, 10000)
        #n_short  = int(t_length / self.dt)
        #freqs_short = xp.fft.rfftfreq(n_short, self.dt)[1:]

        modes = [(2,2)]

        #n = int(t_ref / self.dt)
        delta_t = (2.0*self.dt*self.t_length)/YRSID_SI    
        wave = wave_gen(m1, m2, a1, a2,
                        dist, phi_ref, f_ref, inc, lam,
                        beta, psi, tc, freqs = self.freqs, # 
                        modes=modes, direct=False, fill=True, squeeze=True, length=1024, t_obs_start=tc/YRSID_SI - delta_t, t_obs_end=(tc + 5000.)/YRSID_SI, shift_t_limits=True) # t_obs_start=(2.0*self.dt*self.t_length)/YRSID_SI) # [0] 
        truths = np.array([mu, q, a1, a2, np.cos(inc), lamL, np.sin(betaL), psiL, phi_ref, 0.0])
 
        # NOTE REMOVE THIS
        #wave_min = wave_gen(m1, m2, a1, a2,
        #                dist, phi_ref, f_ref, inc, lam,
        #                beta, psi, tc-1000.0, freqs = self.freqs, # 
        #                modes=modes, direct=False, fill=True, squeeze=True, length=1024,  t_obs_start=tc/YRSID_SI - delta_t, t_obs_end=(tc + 5000.)/YRSID_SI, shift_t_limits=True) #t_obs_start=(2.0*self.dt*self.t_length)/YRSID_SI) # [0] 

        #wave_max = wave_gen(m1, m2, a1, a2,
        #                dist, phi_ref, f_ref, inc, lam,
        #                beta, psi, tc+1000.0, freqs = self.freqs, # 
        #                modes=modes, direct=False, fill=True, squeeze=True, length=1024, t_obs_start=tc/YRSID_SI - delta_t, t_obs_end=(tc + 5000.)/YRSID_SI, shift_t_limits=True) #t_obs_start=(2.0*self.dt*self.t_length)/YRSID_SI) # [0] 
       
        #freq_long = xp.fft.rfftfreq(YRSID_SI, d = 10.0) 
        #wave_long = wave_gen(m1, m2, a1, a2,
        #                dist, phi_ref, f_ref, inc, lam,
        #                beta, psi, tc, freqs = freq_long, # 
        #                modes=modes, direct=False, fill=True, squeeze=True, length=1024, t_obs_start=1.0) # [0] 


        # Load estimates of the noise PSDs for the different channels to whiten the data
        #noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
        #noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, self.freqs.get())

        noise = AnalyticNoise(self.freqs[1:])
        noisevals_A, noisevals_E, noisevals_T = noise.psd(option="A"), noise.psd(option="E"), noise.psd(option="T")

        # TODO implement my own noise on the GPU !!!
        noiseA_cp = xp.asarray(noisevals_A)
        noiseE_cp = xp.asarray(noisevals_E)

        # with noise just for plotting
        #noiseA = sample_noise(xp.array(noisevals_A), self.df, self.dt)#, A_out.shape)
        #noiseE = sample_noise(xp.array(noisevals_E), self.df, self.dt)
        #noiseT = sample_noise(xp.array(noisevals_T), self.df, self.dt)
        #testA = xp.fft.irfft(wave[:,0,:] + noiseA, axis = 1).get()
        #testE = xp.fft.irfft(wave[:,1,:] + noiseE, axis = 1).get()
        #testT = xp.fft.irfft(wave[:,2,:] + noiseT, axis = 1).get()

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        # NOTE Make long waveform to compare
        #n = int(tc/self.dt)  #int(1.0 * YRSID_SI / self.dt)
        #data_freqs = xp.fft.rfftfreq(n, self.dt)[1:]  
        #df_test = data_freqs[2] - data_freqs[1]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      
        # Test another frequency range for the short waveform 
        #data_freqs = xp.linspace(1e-4, 0.05, 2**14) # 2^14 #xp.logspace(-4, 0, 10000)
        #df_test =  data_freqs[2] - data_freqs[1]
      
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Whiten the waveforms
        Afs_white = wave[:,0,1:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseA_cp)
        Efs_white = wave[:,1,1:]*xp.sqrt(4.0*self.df)*self.dt/xp.sqrt(noiseE_cp)

    
        t_shift = xp.asarray(tc) - 2.0*self.dt*self.t_resolution + 5000.0
        shift = xp.exp(1j*2.0*np.pi*self.freqs*t_shift)

        # Inverse transform
        Ats_arr = xp.fft.irfft(xp.c_[xp.zeros(Afs_white.shape[0]), Afs_white]*shift, axis = 1)
        Ets_arr = xp.fft.irfft(xp.c_[xp.zeros(Efs_white.shape[0]) ,Efs_white]*shift, axis = 1)

        # Shift time domain waveform such that the merger is not at the end of the waveform 
        ts_mbhb = xp.c_[Ats_arr, Ets_arr]
      
        return ts_mbhb, truths
