# Code from Michael Katz
# GBGPU/gbgpu/utils/utility.py

import numpy as np
from flow.utils.noisemodel import *
from gbgpu.utils.constants import *

def get_N(amp, f0, Tobs, oversample=1):
    """Determine sampling rate for slow part of FastGB waveform.
    
    Args:
        amp (xp.ndarray): Amplitude of gravitational wave.
        f0 (xp.ndarray): Frequency of gravitational wave in Hz.
        Tobs (double): Observation time in seconds.
        oversample (int, optional): Oversampling factor. This function will return
            ``oversample * N``, if N is the determined sample number. 
            (Default: ``1``). 
    Returns:
        int xp.ndarray: N values for each binary entered.
    
    """

    # make sure they are arrays
    amp = np.atleast_1d(amp)
    f0 = np.atleast_1d(f0)

    # default mult
    mult = 8

    # adjust mult based on observation time
    if (Tobs / YEAR) <= 1.0:
        mult = 1

    elif (Tobs / YEAR) <= 2.0:
        mult = 2

    elif (Tobs / YEAR) <= 4.0:
        mult = 4

    elif (Tobs / YEAR) <= 8.0:
        mult = 8

    # cast for all binaries
    mult = np.full_like(f0, mult, dtype=np.int32)

    N = 32 * mult

    # adjust based on the frequency of the source
    N[f0 >= 0.1] = 1024 * mult[f0 >= 0.1]
    N[(f0 >= 0.03) & (f0 < 0.1)] = 512 * mult[(f0 >= 0.03) & (f0 < 0.1)]
    N[(f0 >= 0.01) & (f0 < 0.3)] = 256 * mult[(f0 >= 0.01) & (f0 < 0.3)]
    N[(f0 >= 0.001) & (f0 < 0.01)] = 64 * mult[(f0 >= 0.001) & (f0 < 0.01)]

    # if a sensitivity curve is available, verify the SNR is not too high
    # if it is, needs more points
    # FIND WHERE I HAVE THE VALUES OF THE CONSTANTS
    fstar = Clight/(Larm*2.0*np.pi)
    tdi_available = True
    if tdi_available:
        fonfs = f0 / fstar

        noise = AnalyticNoise(f0, 'MRDv1')
        SnX = np.sqrt(noise.psd(option="X"))

        #  calculate michelson noise
        Sm = SnX / (4.0 * np.sin(fonfs) * np.sin(fonfs))

        Acut = amp * np.sqrt(Tobs / Sm)

        M = (2.0 ** (np.log(Acut) / np.log(2.0) + 1.0)).astype(int)

        M = M * (M > N) + N * (M < N)
        N = M * (M > N) + N * (M < N)
    else:
        warnings.warn(
            "Sensitivity information not available. The number of points in the waveform will not be determined byt the signal strength without the availability of the Sensitivity."
        )
        M = N

    M[M > 8192] = 8192

    N = M

    # adjust with oversample
    N_out = (N * oversample).astype(int)

    return N_out
