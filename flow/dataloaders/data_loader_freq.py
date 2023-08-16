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
            dr = f['mbhb_freq/coeff_r']
            di = f['mbhb_freq/coeff_i']
            
            # Return data and the true value of the parameter
            

            m1_min = 2340000.0
            m1_max = 2860000.0

            m2_min = 1120000.0
            m2_max = 1370000.0

            chi_min = 0.0
            chi_max = 1.0

            m1 = (dr.attrs['m1'] - m1_min)/(m1_max - m1_min)
            m2 = (dr.attrs['m2'] - m2_min)/(m2_max - m2_min)

            chi1 = (dr.attrs['chi1'] - chi_min)/(chi_max - chi_min)
            chi2 = (dr.attrs['chi2'] - chi_min)/(chi_max - chi_min)

            return dr[:], di[:], m1, m2, chi1, chi2


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filelist)


