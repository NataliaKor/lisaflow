"""
Train the network with the simple chirplet model.
Create data on the fly.
"""
from data_generation.mbhbs.mbhb_model_Lframe import MBHB_gpu

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

from flow_architecture_play import *

#from data_generation.mbhbs.mbhb_model import MBHB_gpu
#from data_generation.mbhbs.mbhb_model_Lframe import MBHB_gpu



# From SWATS https://github.com/Mrpatekful/swats
def switch_optim(optimizer):

    for group in optimizer.param_groups:
       for w in group['params']:
           print('w.grad = ', w.grad) 
           grad = w.grad.data

           beta1, beta2 = group['betas']
           __getstate__()

           # exponential moving average of gradient values
           exp_avg = torch.zeros_like(w.data)
           # exponential moving average of squared gradient values
           exp_avg_sq = torch.zeros_like(w.data)
           # moving average for the non-orthogonal projection scaling
           exp_avg2 = w.new(1).fill_(0)
  
           # decay the first and second moment running average coefficient
           exp_avg.mul_(beta1).add_(1 - beta1, grad)
           exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
  
           denom = exp_avg_sq.sqrt().add_(group['eps'])       
  
           bias_correction1 = 1 - beta1 ** state['step']
           bias_correction2 = 1 - beta2 ** state['step']
           step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

           p = -step_size * (exp_avg / denom)
           p_view = p.view(-1)
           pg = p_view.dot(grad.view(-1))

           if pg != 0:
               # the non-orthognal scaling estimate
               scaling = p_view.dot(p_view) / -pg
               exp_avg2.mul_(beta2).add_(1 - beta2, scaling)
               
               # bias corrected exponential average
               corrected_exp_avg = exp_avg2 / bias_correction2
               
               print(corrected_exp_avg)
               print(scaling)
               # checking criteria of switching to SGD training
               if state['step'] > 1 and corrected_exp_avg.allclose(scaling, rtol=1e-6) and corrected_exp_avg > 0:
                        print('Switch to SGD')
                        #group['phase'] = 'SGD'
                        print('lr = ', corrected_exp_avg.item())
                        #group['lr'] = corrected_exp_avg.item()
                        
    return corrected_exp_avg.item()

def normalise_coeff(coeff, coeff_min, coeff_max):

    return -1.0 + 2.0*(coeff - coeff_min)/(coeff_max - coeff_min)

# Scale all coefficients

  
    return coeff/coeff_maxabs


def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/mbhbs/mbhb_resample_all.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='configs/mbhbs/mbhb_data_radler_sky.yaml',
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

    
    #############################################################################################################################
  
    # Calculate decomposition if it is the first run.
    # Load decomposition if it is subsecuent run with the same setup.
   
    ##############################################################################################################################
   
    # Initialise the class for MBHB waveforms
    mbhb = MBHB_gpu(config, config_data, dtype)
   

    # Number of samples we use for the decomposition
    # TODO the size for the SVD has to larger. Have to write waveforms to the file and calculate SVD from there.
    if config['svd']['estimate']:
        # Initialise a class for MBHB
        Nsamples = 25000
        mbhb.freqwave_AET(Nsamples)
        mbhb_ts = mbhb.timewave_AET()
        Uts, Sts, Vts = torch.svd(torch.as_tensor(mbhb_ts))

        f = h5py.File(config['svd']['root'] + config['svd']['path'], 'w')
        f.create_dataset("S", data=Sts.detach().cpu().numpy())
        f.create_dataset("U", data=Uts.detach().cpu().numpy())
        f.create_dataset("V", data=Vts.detach().cpu().numpy()) 

        # Take only first elements of the coefficients
        Vts_reduce = Vts[:,:num_coeff].type(dtype)
        Sts_reduce = Sts[:num_coeff].type(dtype)

    else:
        fin = h5py.File(config['svd']['root'] + config['svd']['path'], 'r')
        Sts = fin['S']
        Vts = fin['V']

        # Take only first elements of the coefficients
        Vts_reduce = torch.from_numpy(Vts[:,:num_coeff]).type(dtype)
        #Vt = torch.from_numpy(np.transpose(Vts_reduce)).type(dtype)
        Sts_reduce = torch.from_numpy(Sts[:num_coeff]).type(dtype)


    # TODO do we actually need to do this
    # Estimate min and max values for big batch of samples to normalise it
    #coeff, param, _, _, _ = sample_chirp_norm(10000, config_data, t, Sts_reduce, Vts_reduce, dtype)
    #coeff_max  = torch.max(coeff)
    #coeff_min = torch.min(coeff)
    # Check which absolute value is larger for scaling
    #if torch.abs(coeff_max) >= torch.abs(coeff_min):
    #    coeff_maxabs = coeff_max
    #else:
    #    coeff_maxabs = torch.abs(coeff_min)
   

    # EPOCHS
    for j0 in range(number_epochs):

        j = j0 + last_epoch + 1
        print('j = ', j)
 
        flow.train()
        for i in range(number_iterations):

            mbhb.freqwave_AET(batch_size)
            mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)

            noise_samples = torch.randn(mbhb_ts.shape).type(dtype)
            #coeff = torch.matmul(mbhb_ts + noise_samples, (1.0/torch.sqrt(Sts_reduce + 1.0))*Vts_reduce)
         
            coeff = torch.matmul(mbhb_ts + noise_samples, Vts_reduce) 
            #noise_samples = torch.randn(coeff.shape).type(dtype)
 
            coeff_scale = coeff/Sts_reduce 
         
            # Return parameters that correspond to the chosen coefficients
            param = mbhb.get_params()
            #print('param = ', param)
            #print('coeff_scale = ', coeff_scale)

   
            loss = -flow.log_prob(param, coeff_scale).mean()

            new_lr = switch_optim(optimizer)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

 
            if grad_norm_clip_value > 0:
                clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
            optimizer.step()

            # Check if we need to switch the optimiser
            #switch_optim() 

            if anneal_learning_rate:
                scheduler.step()
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
        np.savetxt(config['saving']['save_root']+ 'losses_val' + config['saving']['label'] + '.txt', losses_val)


        with torch.no_grad():

            # Generate waveforms and parameters
            mbhb.freqwave_AET(batch_size)
            mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
            noise_samples = torch.randn(mbhb_ts.shape).type(dtype) 
            coeff = torch.matmul(mbhb_ts + noise_samples, Vts_reduce)  
            coeff_scale = coeff/Sts_reduce
            param = mbhb.get_params()

            loss_val = -flow.log_prob(param, coeff_scale).mean()

            print('loss_val = %.3f' % loss_val)
            losses_val.append(loss_val.tolist())

  
            # Do pp-plot and corner plot after each epoch to see how the distribution of the parameters is converging
            neval = 100    # number of injections
            num_samples = 5000
          
            # 'Real' data
            param_min, param_max, parameter_labels, truths = mbhb.param_ranges()
            wf_true = torch.as_tensor(mbhb.true_data()).type(dtype)
            noise_samples = torch.randn(wf_true.shape).type(dtype)
 
            coeff_true = torch.matmul(wf_true + noise_samples, Vts_reduce)
            #noise_samples = torch.randn(coeff_true.shape).type(dtype)
            coeff_scale_true = coeff_true/Sts_reduce

            #    coeff = torch.matmul(x_wf, torch.from_numpy((1.0/np.sqrt(Sts_reduce))*Vts_reduce).type(dtype)).view(1,-1)

            #coeff_true_norm = scale_coeff(coeff_true, coeff_maxabs)

            # Label for plots
            label = config['plots']['label'] 

            percentiles = np.empty((neval, features_size))
            print('Do corner plot')
            # TODO check what are the values of the truths and coeff_true
            make_cp(flow, j, parameter_labels, np.array(param_min), param_max, coeff_scale_true, truths, label)
         
            print('Do pp plot')
            for idx in range(neval): #tqdm(range(neval)):

                mbhb.freqwave_AET(1)
                mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
                noise_samples = torch.randn(mbhb_ts.shape).type(dtype)
                coeff = torch.matmul(mbhb_ts + noise_samples, Vts_reduce)
                coeff_scale = coeff/Sts_reduce
                param = mbhb.get_params()

                #coeff, param, _, _, _ = sample_chirp_norm(1, config_data, t, Sts_reduce, Vts_reduce, dtype)
                #coeff_norm = scale_coeff(coeff, coeff_maxabs)

                samples = flow.sample(num_samples, coeff_scale).squeeze().cpu().detach().numpy()
                parameters_true = param.cpu().detach().numpy()

                for n in range(features_size):
                    percentiles[idx, n] = stats.percentileofscore(samples[:,n], parameters_true[0,n])

            make_pp(percentiles, parameter_labels, j, label)
            gc.collect()



if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)

