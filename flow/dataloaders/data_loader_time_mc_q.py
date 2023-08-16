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
            d = f['mbhb_time']
          
            # Return data and the true value of the parameter
             
            Mc = d.attrs['Mc']
            q = d.attrs['q']

            return d[:], Mc, q


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filelist)


