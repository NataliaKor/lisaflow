# In the future create base class for fiting and sampling distributions.
# Inherit Galaxy class from the base class.
import numpy as np

import torch
import cupy as cp

from flow.utils.torchutils import *
from flow.distributions.normal import *

from flow.experiments.flow_architecture_density_small import *

class Galaxylog(nn.Module):
    """
     Class for the log Galaxy distribution.
    """
    def __init__(self, config_file):
        super(Galaxylog, self).__init__()
        # Load config file     
        self.config = get_config(config_file)
        # Choose CPU or GPU
        if self.config['gpu'] is not None:
            assert isinstance(self.config['gpu'], int)
            self.dev = f"cuda:{self.config['gpu']}"
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
        param_min, param_max = np.loadtxt('minmax_galaxy_sangria_log.txt')
        self.param_min = self.dtype(param_min)
        self.param_max = self.dtype(param_max)
        
        self.flow = flow

    def sample(self, num_samples):
        """Sample from the extimated distribution.

        Args:
            num_samples: number of samples to draw
        Returns:
            random samples with the corresponding log probabilities
            type: cupy tensor 
        """       
        if self.flow is None:
            raise ValueError(
                    "The values of the weights have to be loaded to the network"
                )
  
        self.flow.eval()
        with torch.no_grad():

            # Check if number of samples is not too large. Generally doesnot work for million samples.
            # Cuda error when tensors are too large.
            if num_samples > 500000:
                num_sampl_small = 100000
                modulo_part = num_samples % num_sampl_small
                integer_part = num_samples // num_sampl_small
                samples, log_prob = self.flow.sample_and_log_prob(num_sampl_small)
                log_prob = torch.unsqueeze(log_prob, 1)
              
                if integer_part > 1:
                    for i in range(0, integer_part-1):
                        samples_temp, log_prob_temp = self.flow.sample_and_log_prob(num_sampl_small)
                        log_prob_temp = torch.unsqueeze(log_prob_temp, 1)
                        samples = torch.vstack([samples, samples_temp])
                        log_prob = torch.vstack([log_prob, log_prob_temp])
                if modulo_part > 0:
                    samples_temp, log_prob_temp = self.flow.sample_and_log_prob(modulo_part)
                    log_prob_temp = torch.unsqueeze(log_prob_temp, 1) 
                    samples = torch.vstack([samples, samples_temp])
                    log_prob = torch.vstack([log_prob, log_prob_temp])
           
            else:

                samples, log_prob = self.flow.sample_and_log_prob(num_samples)
            samples = self.param_min + (samples + 1.0)*(self.param_max - self.param_min)/2.0

            samples_cupy = cp.asarray(samples)
            log_prob_cupy = cp.asarray(log_prob)
        
        return samples_cupy, log_prob_cupy

    # TODO: This has to be different for log Galaxy
    def log_prob(self, inputs_cupy):
        """Calculate log probability for the sample.
           All parameters unnormalised, normalisation performed inside function.
           Amplitude in the actual units and not in log.
 
        Args:
            inputs: sample prom the distribution
            type: cupy tensor
        Returns:
            log probability of the inputs
        """
        inputs_nonorm = torch.as_tensor(inputs_cupy, device = self.dev)
        self.flow.eval()
        with torch.no_grad():
            inputs = 2.0*(inputs_nonorm - self.param_min)/(self.param_max - self.param_min) - 1.0
            log_prob = self.flow.log_prob(inputs)
 
            # Jacobian of the forward transform
            log_prob_norm1_forward = - cp.log(cp.power(10,inputs_nonorm[:,0])) - cp.log(cp.log(10))
            log_prob_norm2_forward = cp.log(8) - cp.log(self.param_max[0] - self.param_min[0]) - \
                                                 cp.log(self.param_max[1] - self.param_min[1]) - \
                                                 cp.log(self.param_max[0] - self.param_min[0])

        log_prob_cupy = cp.asarray(log_prob)
        return log_prob_cupy



