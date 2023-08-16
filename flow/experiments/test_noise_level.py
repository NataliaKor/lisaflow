
import numpy as np
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
from matplotlib import pyplot as plt
from flow.utils.noisemodel import *

#from ldc.lisa.noise import get_noise_model

# Load other noise curve
def load_new_noise(freqs):

    noise = AnalyticNoise(freqs)
    return noise.psd(option="A"), noise.psd(option="E")


# Load noise PSD
def load_noise(freqs):

    noise_evaluator_TDI = pyLISAnoise.initialize_noise(pyLISAnoise.LISAnoiseSciRDv1, TDI='TDIAET', TDIrescaled=False)
    noisevals_A, noisevals_E, noisevals_T = pyLISAnoise.evaluate_noise(pyLISAnoise.LISAnoiseSciRDv1, noise_evaluator_TDI, freqs)

    return noisevals_A, noisevals_E

#def load_noise_ldc(freqs):

#    return get_noise_model("MRDv1", freqs).psd(option="A")
    
# Generate samples of the noise with the defined variance
def sample_noise(psd, df, dt):

   n_real = np.random.normal(loc=0.0, scale=np.sqrt(psd)/(dt*np.sqrt(4.0*df)))
   n_imag = np.random.normal(loc=0.0, scale=np.sqrt(psd)/(dt*np.sqrt(4.0*df)))
 
   return n_real+1j*n_imag


# Estimate psd of the noise using welch method
def estimate_psd(data, dt):
   
    return scipy.signal.welch(data, fs=1.0/dt, window='hanning', nperseg=256*256)



# Define parameters of the observation
dt = 15.0    
Tobs = 31536000.0 
df = 1./Tobs

# Define frequencies to calculate the PSD
N  = int(Tobs / dt)
freqs = np.fft.rfftfreq(N, dt)[1:]

# Create noise in frequency domain
psdA, psdE = load_noise(freqs)
noiseA = sample_noise(psdA, df, dt)

# Create new model of the noise
psdA_new, psdE_new =  load_new_noise(freqs)

# Plot comparison of the noise models
plt.figure()
plt.loglog(freqs, psdA)
plt.loglog(freqs, psdA_new)
plt.savefig('psd_comparison.png')

# Inverse fft
noiseA_time = np.fft.irfft(noiseA)

# Estimate PSD with Welch method
freqs_est, psdA_est = estimate_psd(noiseA_time, dt)

# Plot
plt.figure()
plt.loglog(freqs, psdA)
plt.loglog(freqs, np.abs(noiseA)*np.abs(noiseA))
plt.loglog(freqs_est, psdA_est)
plt.savefig('noise_comparison_dt.png')


# Inverse to time domain and estimate psd
# Plot and compare





