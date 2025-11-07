# Functions that are useful for the MBHB data generation

#from astropy.cosmology import Planck18  
from astropy.cosmology import FlatLambdaCDM




# Convert red shift to distance
def DL(z):

    ldc_cosmo = FlatLambdaCDM(H0=67.1, Om0=0.3175)
    quantity = ldc_cosmo.luminosity_distance(z)
    return quantity.value, quantity.unit


# Normalise parameters to be from -1 to 1
def normalise_par(par, par_min, par_max):

    return 2.0*(par - par_min)/(par_max - par_min) - 1.0


