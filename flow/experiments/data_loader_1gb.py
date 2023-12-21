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
        #self._samples_min = self.samples.min(axis=0)[:-1]
        #self._samples_max = self.samples.max(axis=0)[:-1]

        self._samples_min = 0.0
        self._samples_max = 1.0
           
    def __getitem__(self, index):
        'Generates one sample of data' 
        sampl = 2.0*(self.samples[index,:-1] - self._samples_min)/(self._samples_max - self._samples_min) - 1.0
        return sampl

    def __len__(self):
        'Denotes the total number of samples'
        return self.samples.shape[0]

    def plot_distribution(self):
        figure = corner.corner(self.samples)
        plt.savefig('samples.png')
        plt.close()

    @property
    def samples_min(self):
        return self._samples_min

    @property
    def samples_max(self):
        return self._samples_max

    @property
    def labels(self):
        return ['amp','f0', 'fdot', 'beta', 'lam', 'iota', 'psi', 'phi0']

