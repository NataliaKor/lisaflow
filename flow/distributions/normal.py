import numpy as np
import torch

from flow.distributions.base import Distribution
from flow.utils import torchutils

class StandardNormal(Distribution):
    """
        Multivariate Gaussian distribution with zero mean and unit covariance matrix.
    """
    def __init__(self, shape):
        super().__init__()
        """
        Constructor

        Args:
            shape -- the shape of the input variable (list, tuple or torch.Size) 
        """
        self._shape = torch.Size(shape)

        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _log_prob(self, inputs, context):
        """
         Estimate the log probability for the cost function.
        """
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * \
            torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        """
         Draw samples from the normal distribution.
        """
        if context is None:
            x_sample = torch.randn(num_samples, *self._shape, device=self._log_z.device)
            #print('prob = ', torch.exp(-torch.pow(x_sample,2)/2.0)/torch.sqrt(2.0*torch.pi))
            return x_sample
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape,
                                  device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)


