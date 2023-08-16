"""
This code is based on the paper arXiv:2110.15828 
and the author's implementation https://github.com/VincentStimper/resampled-base-flows
"""

import torch
import torch.nn as nn
import numpy as np

from flow.distributions.base import Distribution
from flow.utils import torchutils


class ResampledGaussian(Distribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix,
    resampled according to a acceptance probability determined by a neural network,
    see arXiv 1810.11428
    """

    def __init__(
        self, 
        shape, 
        acceptance_fn, 
        T = 500, 
        eps = 0.05, 
        trainable=False
        ):
        """
        Constructor
         
        Args:
            shape -- dimension of Gaussian distribution,
            acceptance_fn -- function returning the acceptance probability 
                           (implemented as neural network),
            T -- maximum number of rejections,
            eps -- discount factor in exponential average of Z,
            trainable -- if we learn mean and standard value or not.
        """
        super().__init__()
        self.shape = shape  
        self.acceptance_fn = acceptance_fn # This is a callable function 
        self.T = T
        self.eps = eps
        self.register_buffer("Z", torch.tensor(-1.))
        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                             dtype=torch.float64),
                             persistent=False)

        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *shape))
            self.register_buffer("log_scale", torch.zeros(1, *shape))

    def _log_prob_gaussian(self, inputs_norm):
        """
        Base Gaussian log probability.
        """
        #return  - 0.5 * self.shape * np.log(2 * np.pi) \
        #              - torch.sum(self.log_scale, 1) \
        #              - torch.sum(0.5 * torch.pow(inputs_norm, 2), 1)
        return - self._log_z \
               - torchutils.sum_except_batch(self.log_scale) \
               - 0.5 * torchutils.sum_except_batch(inputs_norm**2) 

    def _log_prob(self, inputs, context=None):
        inputs_norm = (inputs - self.loc) / torch.exp(self.log_scale)
        log_p_gauss = self._log_prob_gaussian(inputs_norm)

        acc = self.acceptance_fn(inputs_norm)
        if self.training or self.Z < 0.:
            eps_ = torch.randn_like(inputs)
            Z_batch = torch.mean(self.acceptance_fn(eps_))
            if self.Z < 0.:
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return log_p

    def _sample(self, num_samples, context=None):
        t = 0
        samples = torch.zeros(num_samples, *self.shape, dtype=self.loc.dtype, device=self.loc.device)
        s = 0
        n = 0
        Z_sum = 0
        if context is None:
            #samples_ = torch.randn(num_samples, *self.shape, device=self._log_z.device)
            context_size = 1
        else:
            context_size = context.shape[0]
        for i in range(self.T):
            samples_ = torch.randn((context_size*num_samples, *self.shape), dtype=self.loc.dtype, device=self.loc.device)
            acc = self.acceptance_fn(samples_)
            if self.training or self.Z < 0.:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    samples[s, :] = samples_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        samples = self.loc + torch.exp(self.log_scale) * samples
       
        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
         num_samples -- Number of samples to draw per batch
         num_batches -- Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            # Get dtype and device
            dtype = self.Z.dtype
            device = self.Z.device
            for i in range(num_batches):
                eps = torch.randn((num_samples, self.shape), dtype=dtype,
                                  device=device)
                acc_ = self.acceptance_fn(eps)
                Z_batch = torch.mean(acc_)
                self.Z = self.Z + Z_batch.detach() / num_batches
