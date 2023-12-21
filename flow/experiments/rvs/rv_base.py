import torch

import cupy as cp

from flow.utils.torchutils import *
from flow.distributions.normal import *

from flow.experiments.flow_architecture_density_small import *

class RV_base(nn.Module):
    """
     Base class for sampling random variables from the fitted distributions.
    """
    def __init__(self, config_file):
        super(RV_base, self).__init__()
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


    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")


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

        # Define base distribution
        distribution = StandardNormal((features_size,)).to(self.dev)
        transform = create_transform(self.config).to(self.dev)

        flow = Flow(transform, distribution).to(self.dev)

        # Define path 
        checkpoint = torch.load(self.config['saving']['save_root'] + self.config['training']['checkpoints'], self.dev)
        flow.load_state_dict(checkpoint['model_state_dict'])

        # Load min and max values to normalise back the distribution
        # Alternatively it can be reestimated through the dataloader class
        # path_minmax = 'minmax_galaxy_sangria.txt' 
        #param_min, param_max = np.loadtxt(path_minmax)
        #self.param_min = self.dtype(param_min)
        #self.param_max = self.dtype(param_max)

        self.flow = flow

    def set_min(self, param_min):
        self.param_min = self.dtype(param_min)
    
    def set_max(self, param_max):
        self.param_max = self.dtype(param_max)
 
    def get_batchs(self, inputs: torch.tensor) -> torch.tensor:
        
        batch_size = int(1e6)
        num_running = inputs.shape[0]
        inds = np.arange(0, num_running, batch_size)
        if inds[-1] != num_running - 1:
            inds = np.concatenate([inds, np.array([num_running - 1])])

        for stind, endind in zip(inds[:-1], inds[1:]):
            yield stind, endind, inputs[stind:endind]


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

        return samples_cupy #, log_prob_cupy


    def log_prob(self, inputs):
        """Calculate log probability under the fitted distribution.

        Args:
            inputs: cupy array, input variables. 
            TODO: check what is the type of the array and convert it accordingly

        Returns:
            A cupy array of shape [input_size], the log probability of the inputs.
        """
        inputs = torch.as_tensor(inputs)
        
        return self._log_prob(inputs)


    def _log_prob(self, inputs):
        raise NotImplementedError()


