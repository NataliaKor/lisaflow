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
        #self._samples_min = self.samples.min(axis=0)[:-3]
        #self._samples_max = self.samples.max(axis=0)[:-3]    
        self._samples_min = self.samples.min(axis=0)[:-1]
        self._samples_max = self.samples.max(axis=0)[:-1]    

        #figure = corner.corner(self.samples[:,:-3])
        figure = corner.corner(self.samples[:,:-1], plot_datapoints=False) 
        plt.savefig('samples_fromStas.png')
        plt.close()

        #figure = corner.corner(2.0*(self.samples[:,:-3] - self._samples_min)/(self._samples_max - self._samples_min) - 1.0)
        #plt.savefig('samples_norm.png')
        #plt.close()
            
    def __getitem__(self, index):
        'Generates one sample of data' 
        #sampl = 2.0*(self.samples[index,:-3] - self._samples_min)/(self._samples_max - self._samples_min) - 1.0
        sampl = 2.0*(self.samples[index,:-1] - self._samples_min)/(self._samples_max - self._samples_min) - 1.0
 
        #sampl = (self.samples[index,:-1] - self._samples_min)/(self._samples_max - self._samples_min)
        #sampl = self.samples[index,:-3]
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
        return ['amp','f0', 'fdot', 'beta', 'lam', 'iota', 'psi', 'phi0']

