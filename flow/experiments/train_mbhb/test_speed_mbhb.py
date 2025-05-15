import cupy as cp
import numpy as np

from bbhx.waveformbuild import BBHWaveformFD
from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.utils.constants import *

from astropy.cosmology import FlatLambdaCDM

import time

# Convert red shift to distance
def DL(z):

    ldc_cosmo = FlatLambdaCDM(H0=67.1, Om0=0.3175)
    quantity = ldc_cosmo.luminosity_distance(z)
    return quantity.value, quantity.unit


response_kwargs = dict(orbits=EqualArmlengthOrbits(use_gpu=True))
tdi_wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True, response_kwargs=response_kwargs)

num_samples = 100
zs = np.zeros(num_samples)

# Read values of parameters from config file
m1 = np.random.uniform(1e5, 1e7, num_samples)
m2 = np.random.uniform(1e5, 1e7, num_samples)
chi1 = np.random.uniform(0., 1., num_samples)
chi2 = np.random.uniform(0., 1., num_samples)
#dist = 15 * 1e9 * PC_SI
z = np.random.uniform(1., 4., num_samples)
phi_ref = np.random.uniform(0., 2.*np.pi, num_samples)
f_ref = zs
inc = np.random.uniform(0., np.pi, num_samples)
lam = np.random.uniform(0., 2.*np.pi, num_samples)
beta = np.random.uniform(0., np.pi, num_samples)
psi = np.random.uniform(0., 2.*np.pi, num_samples)
t_ref = np.random.uniform(1e6, 2e6, num_samples)

modes = [(2,2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]

# Covert redshift to distance
dist = DL(z)[0] * 1e6 * PC_SI


Tobs = YRSID_SI / 12.  # 1 month
dt = 10.0  # sec
N = int(Tobs / dt)
df = 1./Tobs

freqs = cp.arange(1e-4, 0.05, df)
print(freqs.shape)

start = time.time()

AET = tdi_wave_gen(
    m1,
    m2,
    chi1,
    chi2,
    dist,
    phi_ref,
    f_ref,
    inc,
    lam,
    beta,
    psi,
    t_ref,
    length=1024,
    #combine=False,  # TODO: check this
    #direct=False,
    fill=True, #squeeze=True,
    freqs=freqs,
    modes = modes
)

end = time.time()

print((end - start)% 60)
print((end - start) // 60 % 60)
