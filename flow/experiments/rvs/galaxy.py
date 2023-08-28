# In the future create base class for fiting and sampling distributions.
# Inherit Galaxy class from the base class.


class Galaxy(nn.Module):
    """
     Class for the Galaxy distribution.
    """
    def __init__(config_file):

        # Load config file     
        self.config = get_config(config_file)
        # Choose CPU or GPU
        if self.config['gpu'] == 1:
            self.dev = "cuda:0"
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dev = "cpu"
            self.dtype = torch.FloatTensor  

        self.param_min = None
        self.param_max = None

        self.flow = None

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

        # Define base distribution. At the moment there are 2 options: 
        #if self.config['model']['base']['distribution'] == 1:
        distribution = StandardNormal((features_size,)).to(self.dev)

        transform = create_transform(self.config).to(self.dev)
        self.flow = Flow(transform, distribution).to(self.dev)

        # Define path 
        checkpoint = torch.load(self.config['saving']['save_root'] + self.config['training']['checkpoints'])
        flow.load_state_dict(checkpoint['model_state_dict'])

        # Load min and max values to normalise back 
        #filename_ = self.config['samples']['path']
        self.param_min, self.param_max = np.loadtxt('minmax_galaxy_sangria.txt')

    def sample(self, num_samples):
         """Sample from the extimated distribution.

        Args:
            num_samples: number of samples to draw
        Returns:
            randoem samples with the corresponding log probabilities
        """       
        if self.flow is None:
            raise ValueError(
                    "The values of the weights have to be loaded to the network"
                )
        self.flow.eval()
        with torch.no_grad():

            samples, log_prob = self.flow.sample_and_log_prob(num_samples)
            #samples = samples_gpu.squeeze().cpu().detach().numpy()

            for j in range(param_min.shape[0]):
                samples[:,j] = self.param_min[j] + (samples[:,j] + 1.0)*(self.param_max[j] - self.param_min[j])/2.0
        return samples, log_prob


    def log_prob(self, inputs):
        """Calculate log probability for the sample.

        Args:
            inputs: sample prom the distribution
        Returns:
            log probability of the inputs
        """
        log_prob = self.flow.log_prob(inputs)
        return log_prob



