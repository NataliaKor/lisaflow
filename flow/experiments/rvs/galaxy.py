# In the future create base class for fiting and sampling distributions.
# Inherit Galaxy class from the base class.
import numpy as np

import torch
import cupy as cp

from flow.utils.torchutils import *
from flow.distributions.normal import *

from flow.experiments.flow_architecture_density_small import *

from rv_base import RV_base

class Galaxy(RV_base):
    """
        Multivariate Gaussian distribution with zero mean and unit covariance matrix.
    """
    def __init__(self, config_file):
        super(Galaxy, self).__init__(config_file)


    def _log_prob(self, inputs_cupy):
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
            log_prob_norm1_forward = cp.log(cp.log(10)) - cp.log(cp.power(10,inputs_nonorm[:,0]))
            log_prob_norm2_forward = cp.log(8) - cp.log(self.param_max[0] - self.param_min[0]) - \
                                                 cp.log(self.param_max[1] - self.param_min[1]) - \
                                                 cp.log(self.param_max[0] - self.param_min[0])
        log_prob_cupy = cp.asarray(log_prob) + log_prob_norm1_forward + log_prob_norm2_forward
        return log_prob_cupy



