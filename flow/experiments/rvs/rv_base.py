import numpy as np
import argparse
import torch 
from torch.nn.utils import clip_grad_norm_

from flow.utils.torchutils import *
from flow.distributions.normal import *
from flow.distributions.resample import *
from flow.networks.mlp import MLP

from flow.experiments.flow_architecture_density_small import *

from flow.utils.monitor_progress import *
from flow.utils.torchutils import *

from torch.utils.data import DataLoader
from flow.experiments.dataloaders.data_loader_1gb import NPYDataset

import time


def std_get_wrapper(arg):
    return arg

def cuda_get_wrapper(arg):
    return arg.get()

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
            import cupy as cp
            self.xp = cp
            #get_wrapper = cuda_get_wrapper
        else:
            self.dev = "cpu"
            self.dtype = torch.FloatTensor
            import numpy as np
            self.xp = np
            #get_wrapper = std_get_wrapper

        self.param_min = None
        self.param_max = None

        self.flow = None


    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")


    def fit(self):
        """ Fit network to samples.
        """
        # Prepare parameters for training 
        batch_size = self.config['training']['batch_size']
        number_epochs = self.config['training']['epochs']
        number_iterations = self.config['training']['max_iter']
        grad_norm_clip_value = self.config['training']['grad_norm_clip_value']
        anneal_learning_rate =  self.config['training']['anneal_learning_rate']

        # Initialise dataloader
        filename = self.config['samples']['path']
        dataset = NPYDataset(filename)
        # Dataloader for training data
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

        # Record losses
        losses = []

        # Size of the physical parameters
        features_size = self.config['model']['base']['params']

        # Define base distribution. At the moment there are 2 options: 
        if self.config['model']['base']['gaussian'] == 1:
            distribution = StandardNormal((features_size,)).to(self.dev)
        else:
            acceptance_fn = MLP(
            in_shape = [features_size],
            out_shape = [1],
            hidden_sizes = [512, 512],
            activation = F.leaky_relu,
            activate_output = True,
            activation_output = torch.sigmoid)
            distribution = ResampledGaussian((features_size,), acceptance_fn, T = 250).to(self.dev)

        # Define transform
        transform = create_transform(self.config).to(self.dev)
        flow = Flow(transform, distribution).to(self.dev)

        # Set optimisers and schedulers
        # Choose optimiser
        optimizer = optim.Adam(flow.parameters(), lr=self.config['training']['learning_rate'], weight_decay=self.config['training']['weight_decay'])
        # Schedule for learning rate annealing
        if anneal_learning_rate:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.config['training']['num_training_steps'], eta_min=0, last_epoch=-1)
        else:
            scheduler = None

        # Choose to resume training from the previous training results or start fresh
        if self.config['training']['resume']:
            checkpoint = torch.load(self.config['saving']['save_root'] + self.config['training']['checkpoints'])
            flow.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        else:
            last_epoch = -1

        # Load parameter ranges to normalise
        parameter_labels = dataset.labels

        # Load here the labels of parameters
        param_min = dataset.samples_min
        param_max = dataset.samples_max
        np.savetxt(self.config['saving']['save_root'] + 'minmax_' + self.config['saving']['label'] + '.txt', (param_min, param_max))
  
        for j0 in range(number_epochs):

            j = j0 + last_epoch + 1

            flow.train()
            start_epoch = time.time()

            for i, params_cpu in enumerate(loader):

                params = torch.as_tensor(params_cpu).type(self.dtype)
                loss = -flow.log_prob(params).mean()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                if grad_norm_clip_value > 0:
                    clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
                optimizer.step()

            
            end_epoch = time.time()
            print('time per epoch = ', end_epoch - start_epoch)
            print('loss = %.3f' % loss)
            losses.append(loss.tolist())

            # Save checkpoints and loss for every epoch
            checkpoint_path = self.config['saving']['save_root'] + 'checkpoint_{}.pt'.format(str(j+1))
            #if j % 100 == 0:
            torch.save({
                'epoch': j,
                 'model_state_dict': flow.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'loss': loss,}, checkpoint_path)

            torch.save(flow.state_dict(), self.config['saving']['save_root'] + 'checkpoint_model_only.pt')
            #model_scripted = torch.jit.script(flow)
            #model_scripted.save(config['saving']['save_root'] + 'model_scripted.pt')

            np.savetxt(self.config['saving']['save_root'] + 'losses_' + self.config['saving']['label'] + '.txt', losses)

            if anneal_learning_rate:
                scheduler.step()

            # Evaluate and save plots to check
            flow.eval()
            with torch.no_grad():

                # Label for plots
                label = self.config['plots']['label']
                make_cp_density_estimation_minus1(flow, j, parameter_labels, param_min, param_max, label, filename)
                gc.collect()


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

            #samples = torch.zeros((num_samples,))
            #for (stind, endind, samples_batch) in self.get_batchs(samples):
            #    samples[stind: endind] = self.flow.sample()

            # Check if number of samples is not too large. Generally doesnot work for million samples.
            # Cuda error when tensors are too large.
            if num_samples > 500000:
                num_sampl_small = 100000
                modulo_part = num_samples % num_sampl_small
                integer_part = num_samples // num_sampl_small
                samples = self.flow.sample(num_sampl_small)

                if integer_part > 1:
                    for i in range(0, integer_part-1):
                        samples_temp = self.flow.sample(num_sampl_small) 
                        samples = torch.vstack([samples, samples_temp])
                if modulo_part > 0:
                    samples_temp = self.flow.sample(modulo_part)
                    samples = torch.vstack([samples, samples_temp])
            else:
                samples = self.flow.sample(num_samples)

            samples = self._renormalise(samples)

        return self.xp.asarray(samples) 


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

    def _renormalise(self, inputs):
        raise NotImplementedError()

