# Class to sample from the Galaxy.
import torch
import numpy as np
import cupy as cp
from rv_base import RV_base

class Galaxy(RV_base):
    """
        Class for the Galaxy distribution.
    """
    def __init__(self, config_file):
        super(Galaxy, self).__init__(config_file)

    def _log_prob(self, inputs_cupy):
        """Calculate log probability for the sample.
           All parameters unnormalised, normalisation performed inside function.
           Amplitude is passed to a function in log10.
           Order of the parameters [log10(A), sin(beta), lambda]. 
        Args:
            inputs_cupy: samples from the distribution
        Returns:
            log probability of the inputs
        """

        inputs_nonorm = torch.as_tensor(inputs_cupy, device = self.dev)

        self.flow.eval()
        with torch.no_grad():
            
            inputs = 2.0*(inputs_nonorm - self.param_min)/(self.param_max - self.param_min) - 1.0
            
            log_prob = torch.zeros((inputs.shape[0],))
            for (stind, endind, inputs_batch) in self.get_batchs(inputs):
                log_prob[stind: endind] = self.flow.log_prob(inputs_batch)

            # Jacobian of the forward transform
            n_param = self.param_max.size(dim=0)
            log_prob_norm_forward = self.xp.log(self.xp.power(2.,n_param)) - self.xp.sum(self.xp.log(self.param_max - self.param_min))
 
        log_prob_cupy = cp.asarray(log_prob) + log_prob_norm_forward
        return log_prob_cupy

    def _renormalise(self, inputs):
        inputs = self.param_min + (inputs + 1.0)*(self.param_max - self.param_min)/2.0
        return inputs
