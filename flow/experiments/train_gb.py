"""
Train the network with the simple chirplet model.
Create data on the fly.
"""
from data_generation.gbs.gb_model import GB_gpu

import numpy as np
import cupy as cp

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

from flow_architecture import *

# Scale waveforms
def scale_wf(wf, wf_maxabs, dtype):

    return torch.as_tensor(wf/wf_maxabs).type(dtype)


def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/gbs/gb_resample.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='configs/gbs/gb_test.yaml',
                        help='Path to config file specifying parameters of the source when we sample on the fly')
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
    config_data = get_config(args.config_data)

    # Set seed if needed
    seed = None
    if 'seed' in config['training'] and config['training']['seed'] is not None:
        seed = config['training']['seed']
    elif distributed:
        seed = 0
    if seed is not None:
        torch.manual_seed(seed)

    # Prepare training data 
    batch_size = config['training']['batch_size']
    number_epochs = config['training']['epochs']  
    number_iterations = config['training']['max_iter']
    grad_norm_clip_value = config['training']['grad_norm_clip_value']
    anneal_learning_rate =  config['training']['anneal_learning_rate']

    # Record losses
    losses = []
    losses_val = []
 
    # Define number of coefficients and context size after embedding
    num_coeff = config['model']['context']['coeffs']
    context_features_size = config['model']['context']['context_features']
    
    # Size of the physical parameters
    features_size = config['model']['base']['params']

    # Define base distribution. At the moment there are 2 options: 
    if config['model']['base']['gaussian']:
        distribution = StandardNormal((features_size,)).to(dev)
    else:
        acceptance_fn = MLP(
        in_shape = [features_size],
        out_shape = [1],
        hidden_sizes = [512, 512],
        activation = F.leaky_relu,
        activate_output = True,
        activation_output = torch.sigmoid)
        distribution = ResampledGaussian((features_size,), acceptance_fn).to(dev) 
       
    transform = create_transform(config).to(dev)

    # Choose activation function
    if config['model']['embedding']['activation'] == 'elu':
        activation = F.elu
    else:
        activation = F.relu    

    # Define embedding network
    embedding_net = ResidualNet(
            in_features = num_coeff,
            out_features = context_features_size,
            hidden_features = config['model']['embedding']['hidden_features'],
            context_features = None,
            num_blocks = config['model']['embedding']['num_blocks'],
            activation = activation,
            dropout_probability = config['model']['embedding']['dropout'],
            use_batch_norm = config['model']['embedding']['batch_norm'])

    flow = Flow(transform, distribution, embedding_net).to(dev)    

    #########################################################################################################################
    # Set optimisers and schedulers

    # Choose optimiser
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
        #learning_rate = 1.7e-4
        #for g in optimizer.param_groups:
        #    g['lr'] = learning_rate 
    else:
        last_epoch = -1

    
    gb = GB_gpu(config, config_data, dtype)

    # Estimate variable from the ansamble of samples
    # This function has to pass a combination of A and E
    #Ar, Er, Aim, Eim = gb.freqwave_AET(1000)
    A, E = gb.freqwave_AET(1000)
 
    max_Ar_batch = cp.amax(cp.real(A))
    max_Er_batch = cp.amax(cp.real(E))
    max_Aim_batch = cp.amax(cp.imag(A))
    max_Eim_batch  = cp.amax(cp.imag(E))
    #print('max_Ar_batch = ', max_Ar_batch)
    #print('max_Er_batch = ', max_Er_batch)
    #print('max_Aim_batch = ', max_Aim_batch)
    #print('max_Eim_batch = ', max_Eim_batch)

    dt = config_data['tvec']['dt'] 
    maxA = cp.sqrt(max_Ar_batch**2 + max_Aim_batch**2)/cp.sqrt(2.0*dt)
    maxE = cp.sqrt(max_Er_batch**2 + max_Eim_batch**2)/cp.sqrt(2.0*dt)

    print('maxA = ',maxA)
    print('')

    # EPOCHS
    for j0 in range(number_epochs):

        j = j0 + last_epoch + 1
        print('j = ', j)
 
        flow.train()
        for i in range(number_iterations):

            #Ar, Er, Aim, Eim = gb.freqwave_AET(batch_size)
            A, E = gb.freqwave_AET(batch_size)
            param = gb.get_params()

            # Add white noise and scale
            #Ar_scale = scale_wf((Ar + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Er_scale = scale_wf((Er + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #Aim_scale = scale_wf((Aim + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Eim_scale = scale_wf((Eim + cp.random.random_sample(Ar.shape)), maxE, dtype)        
               
            waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype), torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)

            loss = -flow.log_prob(param, waveform).mean()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            if grad_norm_clip_value is not None:
                clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
            optimizer.step()

            if anneal_learning_rate:
                scheduler.step()
            print('i = ', i, 'loss = %.3f' % loss)
            # Output current value of the learning rate
            for param_group in optimizer.param_groups:
                print( param_group['lr'])

        print('loss = %.3f' % loss)
        losses.append(loss.tolist())
        gc.collect()

        # Save checkpoints and loss for every epoch
        checkpoint_path = config['saving']['save_root'] + 'checkpoint_{}.pt'.format(str(j+1))
        torch.save({
           'epoch': j,
           'model_state_dict': flow.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           'loss': loss,}, checkpoint_path)
        np.savetxt(config['saving']['save_root'] + 'losses_' + config['saving']['label'] + '.txt', losses)
        np.savetxt(config['saving']['save_root']+ 'losses_val' + config['saving']['label'] + '.txt', losses_val)

        # Evaluate, and record corner and pp plots
        flow.eval()
        with torch.no_grad():


            # This function has to pass a combination of A and E
            A, E = gb.freqwave_AET(batch_size)
            param = gb.get_params()

            # Add white noise and scale
            #Ar_scale = scale_wf((Ar + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Er_scale = scale_wf((Er + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #Aim_scale = scale_wf((Aim + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Eim_scale = scale_wf((Eim + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #waveform = torch.cat((Ar, Er, Aim, Eim), 1)
            waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype), torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)


            loss_val = -flow.log_prob(param, waveform).mean()

            print('loss_val = %.3f' % loss_val)
            losses_val.append(loss_val.tolist())

    
            # Do pp-plot and corner plot after each epoch to see how the distribution of the parameters is converging
            neval = 100    # number of injections
            num_samples = 1000
          
            # 'Real' data
            param_min, param_max, parameter_labels, truths = gb.param_ranges()
            A, E = gb.true_data()
            # Add white noise and scale
            #Ar_scale = scale_wf((Ar + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Er_scale = scale_wf((Er + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #Aim_scale = scale_wf((Aim + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Eim_scale = scale_wf((Eim + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #waveform = torch.cat((Ar, Er, Aim, Eim), 1)
            waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype), torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)

            # Label for plots
            label = config['plots']['label'] 
            
            percentiles = np.empty((neval, features_size))
            print('Do corner plot')
            # TODO check what are the values of the truths and coeff_true
            make_cp(flow, j, parameter_labels, np.array(param_min), param_max, waveform, truths, label)

            print('Do pp plot')
            for idx in tqdm(range(neval)):

                # This function has to pass a combination of A and E
                A, E = gb.freqwave_AET(1)
                param = gb.get_params()
 
                # Add white noise and scale
                #Ar_scale = scale_wf((Ar + cp.random.random_sample(Ar.shape)), maxA, dtype)
                #Er_scale = scale_wf((Er + cp.random.random_sample(Ar.shape)), maxE, dtype)
                #Aim_scale = scale_wf((Aim + cp.random.random_sample(Ar.shape)), maxA, dtype)
                #Eim_scale = scale_wf((Eim + cp.random.random_sample(Ar.shape)), maxE, dtype)
                #waveform = torch.cat((Ar, Er, Aim, Eim), 1)
                waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype), torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)
             
                samples = flow.sample(num_samples, waveform).squeeze().cpu().detach().numpy()
                parameters_true = param.cpu().detach().numpy()

                for n in range(features_size):
                    percentiles[idx, n] = stats.percentileofscore(samples[:,n], parameters_true[0,n])

            make_pp(percentiles, parameter_labels, j, label)
            gc.collect()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)

