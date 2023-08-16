# Data Loader
from torch.utils.data import Dataset
import numpy as np
import cupy as cp 
import h5py

import corner
from matplotlib import pyplot as plt

# Dataset to use with the dataloader of the npy files with samples
class GalaxyDataset(Dataset):
    
    def __init__(self, filename):
        'Initialisation'
        with h5py.File(filename, "r") as f: 
            load_data = f['gbs_sky_dist'][()]
            self.samples = np.array(load_data.tolist())
            self.samples[:,0] = np.log10(self.samples[:,0])
          
        self._samples_min = self.samples.min(axis=0)
        self._samples_max = self.samples.max(axis=0)
        print('self._samples_min = ', self._samples_min) 
        print('self._samples_max = ', self._samples_max)


        #figure = corner.corner(self.samples)
        #plt.savefig('samples.png')
        #plt.close()

        #figure = corner.corner((self.samples - self._samples_min)/(self._samples_max - self._samples_min))
        figure = corner.corner(2.0*(self.samples - self._samples_min)/(self._samples_max - self._samples_min) - 1.0)
        plt.savefig('samples_galaxy_norm.png')
        plt.close()
            
    def __getitem__(self, index):
        'Generates one sample of data' 
        #sampl = (self.samples[index] - self._samples_min)/(self._samples_max - self._samples_min)
        sampl = 2.0*(self.samples[index] - self._samples_min)/(self._samples_max - self._samples_min) - 1.0
        return sampl

    def __len__(self):
        'Denotes the total number of samples'
        return self.samples.shape[0]

    @property
    def samples_min(self):
        return self._samples_min

    @property
    def samples_max(self):
        return self._samples_max

    @property
    def labels(self):

        return ['amp', 'lambda', 'beta']

