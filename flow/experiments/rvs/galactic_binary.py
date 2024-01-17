# In the future create base class for fitting and sampling distributions.
# TODO: Create base class for different ways to sample.
import torch
from rv_base import RV_base

class GalacticBinary(RV_base):
    """
        Class for the distribution of the single Galactic binary.
    """
    def __init__(self, config_file):
        super(GalacticBinary, self).__init__(config_file)

    def _log_prob(self, inputs_cupy):
        """Calculate log probability for the sample.
           All parameters unnormalised, normalisation performed inside function.
           Parameters of the Galactic Binaries are normalised from 0 to 1.  
           We have to take into account all the Jacobians of the Normalisation transforms.
        Args:
            inputs: samples from the distribution of the single galactic binary. We have to check if the array is on the CPU or on the GPU.
        Returns:
            log probability of the inputs
        """
        # Make variable agnostic to be used on CPU/GPU
        # xp = cp.get_array_module(inputs_cupy)
        inputs = torch.as_tensor(inputs_cupy, device = self.dev)
        self.flow.eval()
        with torch.no_grad():
            inputs = 2.0*(inputs - self.param_min)/(self.param_max - self.param_min) - 1.0   
      
            log_prob = torch.zeros((inputs.shape[0],))
            #log_prob = self.flow.log_prob(inputs)
            for (stind, endind, inputs_batch) in self.get_batchs(inputs):
                log_prob[stind: endind] = self.flow.log_prob(inputs_batch)

            # Jacobian of the forward transform
            #log_prob_norm1_forward = cp.log(cp.log(10)) - cp.log(cp.power(10,inputs_nonorm[:,0])) + \
            #                                              cp.cos(inputs_nonorm[:,1])
            log_prob_norm_forward = self.xp.log(8) - self.xp.log(self.param_max[0] - self.param_min[0]) - \
                                                     self.xp.log(self.param_max[1] - self.param_min[1]) - \
                                                     self.xp.log(self.param_max[0] - self.param_min[0])
        #log_prob_cupy = cp.asarray(log_prob) + log_prob_norm_forward
        return self.xp.asarray(log_prob) + log_prob_norm_forward

    def _renormalise(self, inputs):

        inputs = self.param_min + (inputs + 1.0)*(self.param_max - self.param_min)/2.0
        return inputs                                                                    
