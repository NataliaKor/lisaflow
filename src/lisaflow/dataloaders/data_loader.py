# Data Loader
from torch.utils.data import Dataset
import h5py
import numpy as np
import lisabeta.lisa.lisa as lisa


# Dataset to use with the dataloader of the hdf5 files
class HDF5Dataset(Dataset):
    
    def __init__(self, filelist, path):
        'Initialisation'
        'Label is either train or test' 
        self.filelist = filelist
        self.path = path
                
    def __getitem__(self, index):
        'Generates one sample of data' 
        filename = self.path + self.filelist[index]
      
        with h5py.File(filename, 'r') as f:
            # Name of the dataset
            d = f['mbhb']

            # Return data and the true value of the parameter
            x = d[:]

            m1_min = 2340000.0
            m1_max = 2860000.0

            m2_min = 1120000.0
            m2_max = 1370000.0

            chi_min = 0.0
            chi_max = 1.0

            z_min = 4.5
            z_max = 10.0

            # Transform sky localisation to the LISA reference frame from SSB
            tL_min, tL_max = 24959305.0, 24960683.0
            lamL_min, lamL_max = -np.pi, np.pi
            betL_min, betL_max = -np.pi/2.0, np.pi/2.0
            psiL_min, psiL_max = 0.0, np.pi

            tcL = d.attrs['tcL']
            lamL = d.attrs['lamL']
            betL = d.attrs['betL']
            psiL = d.attrs['psiL']          
            
            tc = (tcL - tL_min)/(tL_max - tL_min)

            beta = (betL - betL_min)/(betL_max - betL_min)
            lambd = (lamL - lamL_min)/(lamL_max - lamL_min)

            m1 = (d.attrs['m1'] - m1_min)/(m1_max - m1_min)
            m2 = (d.attrs['m2'] - m2_min)/(m2_max - m2_min)

            chi1 = (d.attrs['spin1'] - chi_min)/(chi_max - chi_min)
            chi2 = (d.attrs['spin2'] - chi_min)/(chi_max - chi_min)

            phi0 = d.attrs['phi0']/(2.0*np.pi)

            incl = d.attrs['incl']/np.pi
            psi = (psiL - psiL_min)/(psiL_max - psiL_min)
    
            z = (d.attrs['z'] - z_min)/(z_max - z_min)

            return x, beta, lambd, tc, m1, m2, chi1, chi2, incl, psi, phi0, z


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filelist)


