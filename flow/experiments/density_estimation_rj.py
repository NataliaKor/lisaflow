"""
Fit to the samples of the galactic binary
"""
import numpy as np
from matplotlib import pyplot as plt

import argparse
from tqdm import tqdm
from scipy import stats
import h5py

import torch
from torch.nn.utils import clip_grad_norm_

from flow.utils.monitor_progress import *
from flow.utils.torchutils import *
from flow.distributions.normal import *
from flow.distributions.resample import *
from flow.networks.mlp import MLP

from flow_architecture_density import *
from torch.utils.data import DataLoader
from data_loader_rj import NPYDataset 

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/gbs/density.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--resume', type=int, default=1, help='Flag whether to resume training')

    args = parser.parse_args()

    # Choose CPU or GPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        dtype = torch.cuda.FloatTensor
    else:
        dev = "cpu"
        dtype = torch.FloatTensor
    print('device = ', dev)
    cuda = torch.cuda.is_available()

    # Load config
    config = get_config(args.config)

    # Prepare training data 
    batch_size = config['training']['batch_size']
    number_epochs = config['training']['epochs']
    number_iterations = config['training']['max_iter']
    grad_norm_clip_value = config['training']['grad_norm_clip_value']
    anneal_learning_rate =  config['training']['anneal_learning_rate']

    # Initialise dataloader
    filename = config['samples']['path']
    dataset_gb = NPYDataset(filename)
    # Dataloader for training data
    loader = DataLoader(dataset_gb,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    # Record losses
    losses = []

    # Size of the physical parameters
    # num_coeff = feature_size
    features_size = config['model']['base']['params']

    # Define base distribution. At the moment there are 2 options: 
    distribution = StandardNormal((features_size,)).to(dev)
    transform = create_transform(config).to(dev)

    flow = Flow(transform, distribution).to(dev)

    #########################################################################################################################
    # Set optimisers and schedulers

    # Choose optimiser
    #optimizer = optim.SGD(flow.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    optimizer = optim.Adam(flow.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    # Schedule for learning rate annealing
    if anneal_learning_rate:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['training']['num_training_steps'], eta_min=0, last_epoch=-1)
    else:
        scheduler = None

    # TODO save scheduler in the checkpoint and load it
    # Choose to resume training from the previous training results or start fresh
    if config['training']['resume']:
        checkpoint = torch.load(config['saving']['save_root'] + config['training']['checkpoints'])
        flow.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        last_epoch = -1

    # Use this if we want to change the optimiser
    #optimizer = optim.SGD(flow.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    #if anneal_learning_rate:
    #    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['training']['num_training_steps'], eta_min=0, last_epoch=-1)
    #else:
    #    scheduler = None
    
    #le = 0
    #if anneal_learning_rate:
    #    while le < last_epoch:
    #        scheduler.step()

    #gb1 = next(iter(loader))   
     
    # Load here the ranges for parameters
    parameter_labels = dataset_gb.labels

    # Load here the labels of parameters
    param_min = dataset_gb.samples_min
    param_max = dataset_gb.samples_max
    ################################################################
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # In this case there is no validation data!!! 
    # Split data for train and validation in the future!

    # EPOCHS 
    for j0 in range(number_epochs):

        j = j0 + last_epoch + 1

        print('j = ', j)
        flow.train()

        for i, params_cpu in enumerate(loader):

            params = torch.as_tensor(params_cpu).type(dtype)
            # Check if there are no bugs if conditioning is not present
            loss = -flow.log_prob(params).mean()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            if grad_norm_clip_value > 0:
                clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
            optimizer.step()

            # Check if we need to switch the optimiser
            #switch_optim() 
            print('i = ', i, 'loss = %.3f' % loss)
            # Output current value of the learning rate
            for param_group in optimizer.param_groups:
                print( param_group['lr'])

            gc.collect()

        print('loss = %.3f' % loss)
        losses.append(loss.tolist())
        #gc.collect()

        # Save checkpoints and loss for every epoch
        checkpoint_path = config['saving']['save_root'] + 'checkpoint_{}.pt'.format(str(j+1))
        torch.save({
           'epoch': j,
           'model_state_dict': flow.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           'loss': loss,}, checkpoint_path)
        np.savetxt(config['saving']['save_root'] + 'losses_' + config['saving']['label'] + '.txt', losses)
        # Evaluate, and record corner and pp plots

        if anneal_learning_rate:
            scheduler.step()

        flow.eval()
        with torch.no_grad():
            print('Do evaluation plots') 
            # Label for plots
            label = config['plots']['label']

            print('Do corner plot')
            # TODO check what are the values of the truths and coeff_true
            make_cp_density_estimation(flow, j, parameter_labels, param_min, param_max, label)
            # For this purposes we need to create different pp-plot, where we check for the transformed distribution 
            # how many points were created and if they are whithin the countour they should be in

            exit()
            gc.collect()



if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)


