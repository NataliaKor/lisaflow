# In the future create base class for fiting and sampling distributions.
# Inherit Galaxy class from the base class.
import torch
import cupy as cp
from rv_base import RV_base

class GalaxyFFdot(RV_base):
    """
     Class for the Galaxy distribution.
    """
    def __init__(self, config_file):
        super(GalaxyFFdot, self).__init__(config_file)

    def _log_prob(self, inputs_cupy):
        """Calculate log probability for the sample.
           All parameters unnormalised, normalisation performed inside function.
           Order of the parameters [f, fdot].
        Args:
            inputs: samples from the distribution
        Returns:
            log probability of the inputs
        """
        inputs_nonorm = torch.as_tensor(inputs_cupy, device = self.dev)
        self.flow.eval()
        with torch.no_grad():
            inputs = 2.0*(inputs_nonorm - self.param_min)/(self.param_max - self.param_min) - 1.0
            log_prob = self.flow.log_prob(inputs)

            # Jacobian of the forward transform
            inputs_nonorm[:,0] = torch.exp(inputs_nonorm[:,0])
            inputs_nonorm[:,1] = torch.sign(inputs_nonorm[:,1])*torch.pow(10,(inputs_nonorm[:,1]))
            # I am not sure that everything is correct here with the sign!
            log_prob_norm1_forward = cp.log(cp.log(10)) - cp.log(inputs_nonorm[:,0]) - cp.sign(inputs_nonorm[:,1])*cp.log(cp.abs(inputs_nonorm[:,1])) 
            log_prob_norm2_forward = cp.log(8) - cp.log(self.param_max[0] - self.param_min[0]) - \
                                                 cp.log(self.param_max[1] - self.param_min[1]) - \
                                                 cp.log(self.param_max[0] - self.param_min[0])

            log_prob = self.flow.log_prob(inputs)
        log_prob_cupy = cp.asarray(log_prob) + log_prob_norm1_forward + log_prob_norm2_forward
        return log_prob_cupy
