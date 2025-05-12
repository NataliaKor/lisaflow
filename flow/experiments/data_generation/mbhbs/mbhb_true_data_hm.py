import numpy as np
import cupy as xp

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.transform import *
from bbhx.utils.constants import *

from flow.utils.noisemodel import *
from flow.utils.datagenutils_mbhb import *

from astropy.cosmology import FlatLambdaCDM
from flow.utils.datagenutils_mbhb import *

def create_true_data(config_params, freqs):

    df = 1./config_params['tvec']['Tobs']
    dt = config_params['tvec']['dt']

    f_ref   = config_params['default']['f_ref']
    phi_ref = config_params['default']['phi_ref']
    #m1 = config_params['default']['m1']
    #m2 = config_params['default']['m2']
    # Convert mu and q to m1 and m2
    m1 = config_params['default']['mu'] * ((config_params['default']['q'] + 1)**0.2)/config_params['default']['q']**0.6
    m2 = config_params['default']['mu'] * config_params['default']['q']**0.4 * (config_params['default']['q'] + 1)**0.2

    a1 = config_params['default']['a1']
    a2 = config_params['default']['a2']
    z = config_params['default']['z']

    if config_params['frame'] == 'LISA':
        #beta = config_params['default']['beta']
        #lam = config_params['default']['lam']
        #psi = config_params['default']['psi']
        inc = config_params['default']['inc']
        # TODO: here replace the convergence with the code from BBHX
        t_ref, lam, beta, psi = LISA_to_SSB(config_params['default']['t_ref'], config_params['default']['lam'], np.arcsin(config_params['default']['beta']), config_params['default']['psi'], t0=0.0)
    elif config_params['frame'] == 'SSB':
        beta = config_params['default']['beta']
        lam = config_params['default']['lam']
        #the = config_params['default']['the']
        #phi = config_params['default']['phi']
        inc = config_params['default']['inc']
        psi = config_params['default']['psi']
        t_ref = config_params['default']['t_ref']
    else:
        print('No such reference frame')

    # m1 = mu * ((q + 1)**0.2)/q**0.6
    # m2 = mu * q**0.4 * (q + 1)**0.2
    dist = DL(z)[0] * PC_SI * 1e6

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=True)
    modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

    #psi, inc = lisa.AziPolAngleL2PsiIncl(beta, lam, the, phi)

    # Convert angles
    #if sample_in_frame == 'LISA':
    #    tc, lam, beta, psi = lisa.lisatools.ConvertLframeParamsToSSBframe(tcL, lamL, betaL, psiL, constellation_ini_phase=0.)
    #elif sample_in_frame == 'SSB':
    #    tcL, lamL, betaL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(tc, lam, beta, psi, 0.0)  # Check what exactly is zero value
    #else:
    #    print('No such reference frame')

    mbhb = wave_gen(m1, m2, a1, a2, dist, phi_ref, f_ref, 
                    np.arccos(inc), lam, beta, psi, t_ref, 
                    freqs = freqs, modes = modes, direct=False, fill=True, length=1024)

    param_transform = lambda x, k : np.cos(x) if k == 'inc' else np.sin(x) if k == 'beta' else x 
    truths = []
    for key, value in config_params['estimate'].items():
        if value == 1 :
            #temp = param_transform(config_params['default'][key], key) 
            temp = config_params['default'][key]
            truths.append(temp)

    noise = AnalyticNoise(freqs[1:], 'MRDv1')
    noisevals_A, noisevals_E, noisevals_T = noise.psd(option="A"), noise.psd(option="E"), noise.psd(option="T")

    noiseA_cp = xp.asarray(noisevals_A)
    noiseE_cp = xp.asarray(noisevals_E)

    Afs_white = mbhb[:,0,1:]*xp.sqrt(4.0*df)/xp.sqrt(noiseA_cp)
    Efs_white = mbhb[:,1,1:]*xp.sqrt(4.0*df)/xp.sqrt(noiseE_cp)

    Nfreq = freqs.shape[0]
    #print('Nfreq = ', Nfreq)
    t_shift = xp.asarray(t_ref) + 5000.0 +  2.0*dt*Nfreq 
    shift = xp.exp(1j*2.0*np.pi*freqs*t_shift)

    #Ats_arr = xp.fft.irfft(xp.c_[xp.zeros(Afs_white.shape[0]), Afs_white], axis = 1)
    #Ets_arr = xp.fft.irfft(xp.c_[xp.zeros(Efs_white.shape[0]) ,Efs_white], axis = 1)
    Ats_arr = xp.fft.irfft(xp.c_[xp.zeros(Afs_white.shape[0]), Afs_white]*shift, axis = 1)
    Ets_arr = xp.fft.irfft(xp.c_[xp.zeros(Efs_white.shape[0]) ,Efs_white]*shift, axis = 1)

    # Shift time domain waveform such that the merger is not at the end of the waveform 
    ts_mbhb = xp.c_[Ats_arr, Ets_arr]
    print('ts_mbhb.shape = ', ts_mbhb.shape)

    noise_samples = xp.random.normal(size=ts_mbhb.shape)
    print('truths = ', truths)

    return (ts_mbhb + noise_samples), truths



