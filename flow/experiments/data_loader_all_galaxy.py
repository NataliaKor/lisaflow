# Data Loader
from torch.utils.data import Dataset
import numpy as np
import cupy as cp 

import corner
from matplotlib import pyplot as plt

# Dataset to use with the dataloader of the npy files with samples
class NPYDataset(Dataset):
    
    def __init__(self, filename):
        'Initialisation'
        self.samples = np.load(filename)

        # Take log of frequency and amplitude
        self.samples[:,0] = np.log10(self.samples[:,0])
        self.samples[:,3] = np.log10(self.samples[:,3])
        
        # Shifted log for fdot because it can be negative
        self.samples[:,4] = np.log10(self.samples[:,4] + 3.e-15)

        self._samples_min = self.samples.min(axis=0)
        self._samples_max = self.samples.max(axis=0)
        print('self._samples_min = ', self._samples_min)

        #figure = corner.corner(self.samples)
        #plt.savefig('samples_5.png')
        #plt.close()

        #figure = corner.corner((self.samples - self._samples_min)/(self._samples_max - self._samples_min))
        #plt.savefig('samples_norm_5.png')
        #plt.close()
            
    def __getitem__(self, index):
        'Generates one sample of data' 
        sampl = (self.samples[index,:] - self._samples_min)/(self._samples_max - self._samples_min)
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

        return ['amp','beta', 'lam', 'f0', 'fdot']

