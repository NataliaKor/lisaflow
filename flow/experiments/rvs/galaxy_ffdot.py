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
            inputs[:,0] = np.log(inputs_nonorm[:,0])
            inputs[:,1] = np.sign(inpurs_nonorm[:,1])*np.log10(np.abs(inputs_nonorm[:,1]))
            inputs = 2.0*(inputs - self.param_min)/(self.param_max - self.param_min) - 1.0

            log_prob = torch.zeros((inputs.shape[0],))
            for (stind, endind, inputs_batch) in self.get_batchs(inputs):
                log_prob[stind: endind] = self.flow.log_prob(inputs_batch)

            # Jacobian of the forward transform
            # I am not sure that everything is correct here with the sign!
            log_prob_norm1_forward = cp.log(cp.log(10)) - cp.log(inputs[:,0]) - cp.sign(inputs[:,1])*cp.log(cp.abs(inputs[:,1])) 
            log_prob_norm2_forward = cp.log(8) - cp.log(self.param_max[0] - self.param_min[0]) - \
                                                 cp.log(self.param_max[1] - self.param_min[1]) - \
                                                 cp.log(self.param_max[0] - self.param_min[0])
        log_prob_cupy = cp.asarray(log_prob) + log_prob_norm1_forward + log_prob_norm2_forward
        return log_prob_cupy

    def _renormalise(self, inputs):
 
        inputs = self.param_min + (inputs + 1.0)*(self.param_max - self.param_min)/2.0
        inputs[:,0] = torch.exp(inputs[:,0])
        inputs[:,0] = torch.sign(inputs[:,1])*torch.pow(10,(inputs[:,1]))

        return inputs

