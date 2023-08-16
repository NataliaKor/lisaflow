"""
Train the network with the simple chirplet model.
Create data on the fly.
"""
from data_generation.gbs.gb_model_log import GB_gpu

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
from flow.networks.resnet import ConvResidualNet

from flow_architecture_play import *

torch.set_printoptions(precision=10)

# From SWATS https://github.com/Mrpatekful/swats
def switch_optim(optimizer):

    for group in optimizer.param_groups:
       for w in group['params']:
           if w.grad is None:
               continue
           grad = w.grad.data

           beta1, beta2 = group['betas']
           #state = optimizer.__getstate__()
           #state_values = list(state.values())
           step = optimizer.state[optimizer.param_groups[0]["params"][-1]]["step"]
           
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

           bias_correction1 = 1 - beta1 ** step
           bias_correction2 = 1 - beta2 ** step
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

               #print(corrected_exp_avg)
               #print(scaling)
               #print(corrected_exp_avg - scaling)
               # checking criteria of switching to SGD training
              
               if step > 1 and corrected_exp_avg.allclose(scaling, rtol=1e-1) and corrected_exp_avg > 0: # rtol = 1e-6
                        print('Switch to SGD')
                        #group['phase'] = 'SGD'
                        print('lr = ', corrected_exp_avg.item())
                        #group['lr'] = corrected_exp_avg.item()

    #return corrected_exp_avg.item()


# Scale waveforms
def scale_wf(wf, wf_max, wf_min, dtype):

    return torch.as_tensor((wf - wf_min)/(wf_max - wf_min)).type(dtype)


def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/gbs/gb_resample.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='configs/gbs/gb_log.yaml',
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
    #if seed is not None:

    # Prepare training data 
    batch_size = config['training']['batch_size']
    number_epochs = config['training']['epochs']  
    number_iterations =  100 #config['training']['max_iter']
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
    embed_in = 5
    embedding_net = ConvResidualNet(
            in_channels = embed_in,
            out_channels = 1,
            hidden_channels = config['model']['embedding']['hidden_features'],
            context_channels = None,
            num_blocks = config['model']['embedding']['num_blocks'],
            dropout_probability = config['model']['embedding']['dropout'],
            use_batch_norm = config['model']['embedding']['batch_norm'],
            kernel_size = (1,3)) # 3

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
 
    max_Ar_batch = 10000.0 #cp.amax(cp.real(A))
    max_Er_batch = 10000.0 #cp.amax(cp.real(E))
    max_Aim_batch = 10000.0 #cp.amax(cp.imag(A))
    max_Eim_batch  = 10000.0 #cp.amax(cp.imag(E))

    max_batch = 10000.0

    min_Er_batch = -10000.0 #cp.amin(cp.real(E))
    min_Aim_batch = -10000.0 #cp.amin(cp.imag(A))
    min_Eim_batch  = -10000.0 #cp.amin(cp.imag(E))
    min_Ar_batch = -10000.0 #cp.amin(cp.real(A))

    print('max_Ar_batch = ', max_Ar_batch)
    print('min_Ar_batch = ', min_Ar_batch)
    print('max_Er_batch = ', max_Er_batch)   
    print('max_Aim_batch = ', max_Aim_batch)
    print('max_Eim_batch = ', max_Eim_batch)
    print('min_Er_batch = ', min_Er_batch)   
    print('min_Aim_batch = ', min_Aim_batch)
    print('min_Eim_batch = ', min_Eim_batch)
   
    dt = config_data['tvec']['dt'] 
    #maxA = cp.sqrt(max_Acr_batch**2 + max_Aim_batch**2)/cp.sqrt(2.0*dt)
    #maxE = cp.sqrt(max_Er_batch**2 + max_Eim_batch**2)/cp.sqrt(2.0*dt)

    #print('maxA = ',maxA)
    #print('maxE = ',maxE)
    #print('')

    Amax = torch.as_tensor(10**config_data['limits']['max']['amp'], device = dev)

    # Add frequencies as one of the channels
    freqs = (gb.get_freqs() - config_data['limits']['min']['fvec'])/(config_data['limits']['max']['fvec'] - config_data['limits']['min']['fvec'])
    freqs_arr = torch.as_tensor(cp.tile(freqs,(batch_size, 1))).type(dtype)
    freqs_one = torch.as_tensor(freqs).view(1,-1).type(dtype)

    # EPOCHS
    for j0 in range(number_epochs):

        j = j0 + last_epoch + 1
        print('j = ', j)
 
        flow.train()
        for i in range(number_iterations):

            #Ar, Er, Aim, Eim = gb.freqwave_AET(batch_size)
            A, E = gb.freqwave_AET(batch_size)
            param = gb.get_params()
            #print('param = ', param)
            #print('A = ', A)
            #print('E = ', E)

            # Add white noise and scale
            #Ar_scale = scale_wf(cp.real(A), max_Ar_batch, min_Ar_batch, dtype)
            #Er_scale = scale_wf(cp.real(E), max_Er_batch, min_Er_batch, dtype)
            #Aim_scale = scale_wf(cp.imag(A), max_Aim_batch, min_Aim_batch, dtype)
            #Eim_scale = scale_wf(cp.imag(E), max_Eim_batch, min_Eim_batch, dtype)        
        
       
            waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.real(E)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.imag(A)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.imag(E)/max_batch).type(dtype), freqs_arr), 1)
            #waveform = torch.cat((torch.as_tensor(Ar_scale).type(dtype), torch.as_tensor(Er_scale).type(dtype), torch.as_tensor(Aim_scale).type(dtype), torch.as_tensor(Eim_scale).type(dtype)), 1)
            #print('waveform = ', waveform)
           
            waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))
            print('param.shape = ', param.shape)
            print('waveform_cnn.shape = ', waveform_cnn.shape)    
            loss = -flow.log_prob(param, waveform_cnn).mean()
            
            switch_optim(optimizer)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            if grad_norm_clip_value > 0:
                clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
            optimizer.step()

            #if anneal_learning_rate:
            #    scheduler.step()
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

        if anneal_learning_rate:
            scheduler.step()

        # Evaluate, and record corner and pp plots
        flow.eval()
        with torch.no_grad():


            # This function has to pass a combination of A and E
            A, E = gb.freqwave_AET(batch_size)
            param = gb.get_params()
            #Ar_scale = scale_wf(cp.real(A), max_Ar_batch, min_Ar_batch, dtype)
            #Er_scale = scale_wf(cp.real(E), max_Er_batch, min_Er_batch, dtype)
            #Aim_scale = scale_wf(cp.imag(A), max_Aim_batch, min_Aim_batch, dtype)
            #Eim_scale = scale_wf(cp.imag(E), max_Eim_batch, min_Eim_batch, dtype)

            # Add white noise and scale
            #Ar_scale = scale_wf((Ar + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Er_scale = scale_wf((Er + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #Aim_scale = scale_wf((Aim + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Eim_scale = scale_wf((Eim + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #waveform = torch.cat((Ar, Er, Aim, Eim), 1)

            waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.real(E)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.imag(A)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.imag(E)/max_batch).type(dtype), freqs_arr), 1)
            #waveform = torch.cat((torch.as_tensor(cp.real(A)).type(dtype), torch.as_tensor(cp.real(E)).type(dtype), torch.as_tensor(cp.imag(A)).type(dtype), torch.as_tensor(cp.imag(E)).type(dtype)), 1)
            #waveform = torch.cat((torch.as_tensor(Ar_scale).type(dtype), torch.as_tensor(Er_scale).type(dtype), torch.as_tensor(Aim_scale).type(dtype), torch.as_tensor(Eim_scale).type(dtype)), 1)
            waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))

            loss_val = -flow.log_prob(param, waveform_cnn).mean()

            print('loss_val = %.3f' % loss_val)
            losses_val.append(loss_val.tolist())

    
            # Do pp-plot and corner plot after each epoch to see how the distribution of the parameters is converging
            neval = 100    # number of injections
            num_samples = 1000
          
            # 'Real' data
            param_min, param_max, parameter_labels, truths = gb.param_ranges()
            A, E = gb.true_data()
            #Ar_scale = scale_wf(cp.real(A), max_Ar_batch, min_Ar_batch, dtype)
            #Er_scale = scale_wf(cp.real(E), max_Er_batch, min_Er_batch, dtype)
            #Aim_scale = scale_wf(cp.imag(A), max_Aim_batch, min_Aim_batch, dtype)
            #Eim_scale = scale_wf(cp.imag(E), max_Eim_batch, min_Eim_batch, dtype)

            # Add white noise and scale
            #Ar_scale = scale_wf((Ar + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Er_scale = scale_wf((Er + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #Aim_scale = scale_wf((Aim + cp.random.random_sample(Ar.shape)), maxA, dtype)
            #Eim_scale = scale_wf((Eim + cp.random.random_sample(Ar.shape)), maxE, dtype)
            #waveform = torch.cat((Ar, Er, Aim, Eim), 1)

            print('cp.real(A).shape = ', cp.real(A).shape)
            print('freqs_one.shape = ', freqs_one.shape)
            waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.real(E)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.imag(A)/max_batch).type(dtype), 
                                  torch.as_tensor(cp.imag(E)/max_batch).type(dtype), freqs_one), 1)
            #waveform = torch.cat((torch.as_tensor(cp.real(A)).type(dtype), torch.as_tensor(cp.real(E)).type(dtype), torch.as_tensor(cp.imag(A)).type(dtype), torch.as_tensor(cp.imag(E)).type(dtype)), 1)
            #waveform = torch.cat((torch.as_tensor(Ar_scale).type(dtype), torch.as_tensor(Er_scale).type(dtype), torch.as_tensor(Aim_scale).type(dtype), torch.as_tensor(Eim_scale).type(dtype)), 1)
            waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))

            # Label for plots
            label = config['plots']['label'] 
            
            percentiles = np.empty((neval, features_size))
            print('Do corner plot')
            # TODO check what are the values of the truths and coeff_true

            make_cp_as(flow, j, parameter_labels, np.array(param_min), param_max, waveform_cnn, truths, label, Amax)

            print('Do pp plot')
            for idx in tqdm(range(neval)):

                # This function has to pass a combination of A and E
                A, E = gb.freqwave_AET(1)
                #Ar_scale = scale_wf(cp.real(A), max_Ar_batch, min_Ar_batch, dtype)
                #Er_scale = scale_wf(cp.real(E), max_Er_batch, min_Er_batch, dtype)
                #Aim_scale = scale_wf(cp.imag(A), max_Aim_batch, min_Aim_batch, dtype)
                #Eim_scale = scale_wf(cp.imag(E), max_Eim_batch, min_Eim_batch, dtype)
                param = gb.get_params()
 
                # Add white noise and scale
                #Ar_scale = scale_wf((Ar + cp.random.random_sample(Ar.shape)), maxA, dtype)
                #Er_scale = scale_wf((Er + cp.random.random_sample(Ar.shape)), maxE, dtype)
                #Aim_scale = scale_wf((Aim + cp.random.random_sample(Ar.shape)), maxA, dtype)
                #Eim_scale = scale_wf((Eim + cp.random.random_sample(Ar.shape)), maxE, dtype)
                #waveform = torch.cat((Ar, Er, Aim, Eim), 1)
                #waveform = torch.cat((torch.as_tensor(Ar_scale).type(dtype), torch.as_tensor(Er_scale).type(dtype), torch.as_tensor(Aim_scale).type(dtype), torch.as_tensor(Eim_scale).type(dtype)), 1)
                waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype), 
                                      torch.as_tensor(cp.real(E)/max_batch).type(dtype), 
                                      torch.as_tensor(cp.imag(A)/max_batch).type(dtype), 
                                      torch.as_tensor(cp.imag(E)/max_batch).type(dtype), freqs_one), 1)

                waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))
         
                samples = flow.sample(num_samples, waveform_cnn).squeeze().cpu().detach().numpy()
                parameters_true = param.cpu().detach().numpy()

                for n in range(features_size):
                    percentiles[idx, n] = stats.percentileofscore(samples[:,n], parameters_true[0,n])

            make_pp(percentiles, parameter_labels, j, label)
            gc.collect()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)

