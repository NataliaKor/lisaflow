import numpy as np
import torch

from flow.distributions.base import Distribution
from flow.utils import torchutils

class UniformUnit(Distribution):
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

        # An empty tensor, to get a device from it
        self.register_buffer("_dev", torch.empty([1]))

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
        # I am not sure if this is correct
        return torch.zeros(inputs.shape[0], device=inputs.device)

    def _sample(self, num_samples, context):
        """
         Draw samples from the normal distribution.
        """
        if context is None:
            x_sample = torch.rand(num_samples, *self._shape, device=self._dev.device)
            return x_sample
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.rand(context_size * num_samples, *self._shape,
                                  device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._dev.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)


