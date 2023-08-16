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
         
            m1_load = dr.attrs['m1']
            m2_load = dr.attrs['m2']

            q = m1_load/m2_load
            Mc = (m1_load + m2_load)*(q/(1+q)**2)**0.6

            q_min = m1_min/m2_max
            q_max = m1_max/m2_min

            q_minmin = m1_min/m2_min
            q_maxmax = m1_max/m2_max

            Mc_min = (m1_min + m2_min)*(q_minmin/(1+q_minmin)**2)**0.6
            Mc_max = (m1_max + m2_max)*(q_maxmax/(1+q_maxmax)**2)**0.6

            m1 = (q - q_min)/(q_max - q_min)
            m2 = (Mc - Mc_min)/(Mc_max - Mc_min)

            return dr[:], di[:], m1, m2


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filelist)


