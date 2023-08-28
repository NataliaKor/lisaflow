# In the future create base class for fiting and sampling distributions.
# Inherit Galaxy class from the base class.
import numpy as np

import torch

from flow.utils.torchutils import *
from flow.distributions.normal import *

from flow.experiments.flow_architecture_density_small import *


class Galaxy(nn.Module):
    """
     Class for the Galaxy distribution.
    """
    def __init__(self, config_file):
        super(Galaxy, self).__init__()
        # Load config file     
        self.config = get_config(config_file)
        # Choose CPU or GPU
        if self.config['gpu'] == 1:
            self.dev = "cuda:0"
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dev = "cpu"
            self.dtype = torch.FloatTensor  

        self.param_min = None
        self.param_max = None

        self.flow = None

    def fit(self):
        """ Fit network to samples.
        """
        print('Not implemented') 
        # Load min and max values to normalise back 


    def load_fit(self):
        """Load network that has been already fit to the distribution. 
        """

        # Size of the physical parameters
        features_size = self.config['model']['base']['params']

        # Define base distribution. At the moment there are 2 options: 
        #if self.config['model']['base']['distribution'] == 1:
        distribution = StandardNormal((features_size,)).to(self.dev)

        transform = create_transform(self.config).to(self.dev)
        flow = Flow(transform, distribution).to(self.dev)

        # Define path 
        checkpoint = torch.load(self.config['saving']['save_root'] + self.config['training']['checkpoints'])
        flow.load_state_dict(checkpoint['model_state_dict'])

        # Load min and max values to normalise back 
        #filename_ = self.config['samples']['path']
        param_min, param_max = np.loadtxt('minmax_galaxy_sangria.txt')
        self.param_min = self.dtype(param_min)
        self.param_max = self.dtype(param_max)

        self.flow = flow

    def sample(self, num_samples):
        """Sample from the extimated distribution.

        Args:
            num_samples: number of samples to draw
        Returns:
            randoem samples with the corresponding log probabilities
        """       
        if self.flow is None:
            raise ValueError(
                    "The values of the weights have to be loaded to the network"
                )
        self.flow.eval()
        with torch.no_grad():

            samples, log_prob = self.flow.sample_and_log_prob(num_samples)
            #samples = samples_gpu.squeeze().cpu().detach().numpy()

            samples = self.param_min + (samples + 1.0)*(self.param_max - self.param_min)/2.0
 
        return samples, log_prob


    def log_prob(self, inputs):
        """Calculate log probability for the sample.

        Args:
            inputs: sample prom the distribution
        Returns:
            log probability of the inputs
        """
        self.flow.eval()
        with torch.no_grad():
            inputs = 2.0*(inputs - self.param_min)/(self.param_max - self.param_min) - 1.0
            log_prob = self.flow.log_prob(inputs)
        return log_prob



