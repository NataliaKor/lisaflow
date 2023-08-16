import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from flow.baseflows.base import *
from flow.distributions.normal import *
from flow.flowtransforms.lu import *
from flow.flowtransforms.base import *
from flow.flowtransforms.permutations import *
from flow.flowtransforms.coupling import *
from flow.flowtransforms.normalization import BatchNorm
from flow.networks import ResidualNet
from flow.utils import *

#from  data_loader import HDF5Dataset
import h5py
import gc
import fnmatch


def create_batchnorm_transform(features):

    #return BatchNorm_rnvp(features)
    return BatchNorm(features, eps=1e-10, momentum=0.5)

def create_linear_transform(features):

    return CompositeTransform([
        RandomPermutation(features=features),
        LULinear(features, identity_init=True)])

def create_base_transform(i,config):
    # 'rq-coupling'
    features = config['model']['base']['params']
    hidden_features = config['model']['transform']['hidden_features']
    num_transform_blocks = config['model']['transform']['num_blocks']  
    dropout_probability = config['model']['transform']['dropout']
    use_batch_norm = config['model']['transform']['batch_norm'] # 1 if true
    num_bins = config['model']['transform']['num_bins']
    tail_bound = config['model']['transform']['tail_bound']  # Box is on [-bound, bound]^2
    apply_unconditional_transform = config['model']['transform']['unconditional_transform'] # Whether to unconditionally transform \'identity\' features in coupling layer.

    if config['model']['transform']['activation'] == 'relu':
        activation = F.relu
    else:
        activation = F.elu

    return PiecewiseRationalQuadraticCouplingTransform(
            mask=create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_blocks=num_transform_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        ),
        num_bins=num_bins,
        tails='linear',
        tail_bound=tail_bound,
        apply_unconditional_transform=apply_unconditional_transform
    )

# Rewrite such that parameters are read from the configuration file
def create_base_transform_affine(i, config):
        #'affine-coupling'
        features = config['model']['base']['params']
        #hidden_features = 512 # 256
        #num_transform_blocks = 6
        #dropout_probability = 0.05
        #use_batch_norm = 1 # 1 if true

        hidden_features = config['model']['affine']['hidden_features'] # 512
        num_transform_blocks = config['model']['affine']['num_blocks']# 6
        dropout_probability = config['model']['affine']['dropout'] # 0.2
        use_batch_norm = config['model']['affine']['batch_norm'] # 1 # 1 if true

        if config['model']['transform']['activation'] == 'relu':
            activation = F.relu
        else:
            activation = F.elu

     
        return  AffineCouplingTransform(
                mask=create_alternating_binary_mask(features, even=(i % 2 == 0)),
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=num_transform_blocks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm
                )
         )


def create_transform(config):
    num_flow_steps = config['model']['flow']['num_flow_steps'] 
    features = config['model']['base']['params']
    transform = CompositeTransform([
        CompositeTransform([
            create_linear_transform(features),
            create_base_transform(i, config),    
            #create_batchnorm_transform(features), 
            create_base_transform_affine(i, config),
            #create_batchnorm_transform(features)
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform(features)
    ])
    return transform




