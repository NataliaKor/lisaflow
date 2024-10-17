# In the future create base class for fiting and sampling distributions.
# Inherit Galaxy class from the base class.
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
            log_prob_norm1_forward = cp.log(cp.log(10)) - cp.log(cp.power(10,inputs_nonorm[:,0])) + \
                                                          cp.cos(inputs_nonorm[:,1]) 
            log_prob_norm2_forward = cp.log(8) - cp.log(self.param_max[0] - self.param_min[0]) - \
                                                 cp.log(self.param_max[1] - self.param_min[1]) - \
                                                 cp.log(self.param_max[2] - self.param_min[2])
        log_prob_cupy = cp.asarray(log_prob) + log_prob_norm1_forward + log_prob_norm2_forward
        return log_prob_cupy

    def _renormalise(self, inputs):

        inputs = self.param_min + (inputs + 1.0)*(self.param_max - self.param_min)/2.0
        return inputs
