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
            # Name of the datase
          
            d = f['mbhb_time']

            # Return data and the true value of the parameter
            x = d[:]

            m1_min = 2340000.0
            m1_max = 2860000.0

            m2_min = 1120000.0
            m2_max = 1370000.0

            chi_min = 0.0
            chi_max = 1.0

            m1 = (d.attrs['m1'] - m1_min)/(m1_max - m1_min)
            m2 = (d.attrs['m2'] - m2_min)/(m2_max - m2_min)

            chi1 = (d.attrs['chi1'] - chi_min)/(chi_max - chi_min)
            chi2 = (d.attrs['chi2'] - chi_min)/(chi_max - chi_min)

            return x, m1, m2, chi1, chi2


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filelist)


