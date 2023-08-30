import numpy as np



# Expression for the exponential chirplet
def f_exp_chirp(t, f0, k, a):

    return a*np.exp(-pow((t-12*3600.0)/10000,2))*np.sin(2.0*np.pi*f0*((k**t-1)/np.log(k)))

# Normalised chirplet model
def chirplet_normalised(t, Q, t0, f0, d):
    A = ((8 * np.pi * f0**2) / (Q**2))**(1. / 4)
    paren_term = f0 * (t - t0) + 0.5 * d * (t - t0)**2
    cos_arg = 2 * np.pi * paren_term
    exp_arg = - ((2 * np.pi * f0) / Q)**2 * (t - t0)**2
    return A * np.exp(exp_arg) * np.cos(cos_arg)


