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
        samples = np.load(filename)

        samples[:,0] = np.log(samples[:,0])
        samples[:,1] = np.sign(samples[:,1])*np.log10(np.abs(samples[:,1]))

        self._samples_min = samples.min(axis=0)
        self._samples_max = samples.max(axis=0)
       
        self.samples = 2.0*(samples[index,:] - self._samples_min)/(self._samples_max - self._samples_min) - 1.0
 
        #figure = corner.corner(self.samples)
        #plt.savefig('samples_5.png')
        #plt.close()

        #figure = corner.corner((self.samples - self._samples_min)/(self._samples_max - self._samples_min))
        #plt.savefig('samples_norm_2.png')
        #plt.close()
            
    def __getitem__(self, index):
        'Generates one sample of data' 
        return self.samples[index,:]

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

        return ['f0', 'fdot']

