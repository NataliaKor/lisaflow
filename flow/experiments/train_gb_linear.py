"""
Train the network with the simple chirplet model.
Create data on the fly.
"""
from data_generation.gbs.gb_model_log_wf import GB_gpu

import numpy as np
import cupy as cp

from matplotlib import pyplot as plt

import argparse
from tqdm import tqdm
from scipy import stats
import h5py

import torch
from torch.nn.utils import clip_grad_norm_

from gbgpu.utils.constants import *

from flow.utils.monitor_progress import *
from flow.utils.torchutils import *
from flow.distributions.normal import *
from flow.distributions.resample import *
from flow.networks.mlp import MLP
from flow.networks.resnet import ConvResidualNet, ConvResNet1D, Bottleneck, BasicBlock, ConvResidualNet_VB
from flow.networks.densenet import DenseNet

from flow.utils.gb_freq_number import get_N

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
          
               # checking criteria of switching to SGD training
                
               if step > 1 and corrected_exp_avg[0].isclose(scaling, atol=1e-4) and corrected_exp_avg > 0: # rtol = 1e-6
                        print('Switch to SGD')
                        #group['phase'] = 'SGD'
                        print('lr = ', corrected_exp_avg.item())
                        #group['lr'] = corrected_exp_avg.item()
               
    #return corrected_exp_avg.item()


# Scale waveforms
def scale_wf(wf, wf_max, wf_min, dtype):

    return torch.as_tensor((wf - wf_min)/(wf_max - wf_min)).type(dtype)

def norm_wf(wf, wf_scale, dtype):
    return torch.as_tensor(wf/wf_scale).type(dtype)

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/gbs/gb_resample_linear.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='configs/gbs/gb_VB_3.yaml',
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
    #seed = None
    #if 'seed' in config['training'] and config['training']['seed'] is not None:
    #    seed = config['training']['seed']
    #elif distributed:
    #    seed = 0
    #if seed is not None:

    ###############################################################################
    # This is in the case if what to load all galactic binaries and run for all of them
    ##############################################################################
    Tobs = YEAR * config_data['tvec']['Tobs']
    print('Tobs = ', Tobs)
 
    #training_file = 'sangria/LDC2_sangria_training_v2.h5'
    #fid = h5py.File(training_file)
    #names = fid["sky/vgb/cat"].dtype.names
    #units = [(k, fid['sky/vgb/cat'].attrs.get(k)) for k in names]
    #units = dict(units)
    #params = [fid["sky/vgb/cat"][name] for name in names]
    #fid.close()

    #amp = params[1][:,0]
    #f0 = params[9][:,0]
    amp = config_data['default']['amp']
    f0 =  config_data['default']['f0'] 
    print('Number of frequencies for a chosen set of parameters:')
    num_f = get_N(amp, f0, Tobs, oversample=1)
    print('num_f = ', num_f[0])
    
    ###############################################################################

    # Prepare training data 
    batch_size = config['training']['batch_size']
    number_epochs = config['training']['epochs']  
    number_iterations =  config['training']['max_iter']
    grad_norm_clip_value = config['training']['grad_norm_clip_value']
    anneal_learning_rate =  config['training']['anneal_learning_rate']

    # Record losses
    losses = []
    losses_val = []

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

    # Define number of channels
    embed_in =  4

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

    # Define embedding network
    #embedding_net = ConvResidualNet_VB(
    #        in_channels = embed_in,
    #        out_channels = 1,
    #        hidden_channels = config['model']['embedding']['hidden_features'],
    #        context_channels = None,
    #        num_blocks = config['model']['embedding']['num_blocks'],
    #        dropout_probability = config['model']['embedding']['dropout'],
    #        use_batch_norm = config['model']['embedding']['batch_norm'],
    #)
    #embed_in = 5 # 4

    #depth = 16
    #growth_rate = 6
    # Get densenet configuration
    #if (depth - 4) % 3:
    #    raise Exception('Invalid depth')
    #block_config = [(depth - 4) // 6 for _ in range(3)]

    #embedding_net = DenseNet(
    #    embed_in = embed_in,
    #    growth_rate = 10,
    #    block_config = block_config,
    #    num_init_features = growth_rate*2,
    #    num_classes = config['model']['context']['context_features'],
    #    small_inputs = False,
    #    efficient = True,
    #)
    #embedding_net = ConvResNet1D(
    #        block = BasicBlock, # Bottleneck, 
    #        layers = [3, 2, 2, 3], # 16 layers = [3, 4, 23, 3],
    #        embed_in = embed_in,
    #        num_classes = config['model']['context']['context_features']
    #)
    #embedding_net = (8, num_classes, growth_rate=12,
    #             reduction=0.5, bottleneck=True, dropRate=0.0)
 

    flow = Flow(transform, distribution, embedding_net).to(dev)    
    #flow = Flow(transform, distribution).to(dev)

    #########################################################################################################################
    # Set optimisers and schedulers

    #checkpoint = torch.load(config['saving']['save_root'] + config['training']['checkpoints'])
    #flow.load_state_dict(checkpoint['model_state_dict'])

    # Choose optimiser
    optimizer = optim.Adam(flow.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    #optimizer = optim.SGD(flow.parameters(), lr=config['training']['learning_rate'])
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

    
    gb = GB_gpu(config, config_data, f0, num_f[0], dtype)
    # Create first batch of the waveforms.
    # This is done to estimate mean and std for the distribution of parameters
    # and also for wavefoms to normalise them.
    #A, E = gb.freqwave_AET(500000)
    gb.sample_from_prior(10000, 0)
    #gb.initialise_freq_vector()
    A, E = gb.create_waveform(0) 
 
    plt.figure()
    plt.plot(np.abs(A[1,:].get()))
    
    plt.savefig('sigA.png')
    plt.close()
    plt.figure()
    plt.plot(np.abs(E[1,:].get()))

    plt.savefig('whiA.png')
    plt.close()
 
    max_Ar_batch = cp.amax(cp.abs(cp.real(A)))
    max_Er_batch = cp.amax(cp.abs(cp.real(E)))
    max_Aim_batch = cp.amax(cp.abs(cp.imag(A)))
    max_Eim_batch  = cp.amax(cp.abs(cp.imag(E)))

    #max_batch = 2000.0 # 150.0 # 100.0

    #min_Er_batch = cp.amin(cp.real(E))
    #min_Aim_batch = cp.amin(cp.imag(A))
    #min_Eim_batch  = cp.amin(cp.imag(E))
    #min_Ar_batch = cp.amin(cp.real(A))

    #print('max_Ar_batch = ', max_Ar_batch)
    #print('min_Ar_batch = ', min_Ar_batch)
    #print('max_Er_batch = ', max_Er_batch)   
    #print('max_Aim_batch = ', max_Aim_batch)
    #print('max_Eim_batch = ', max_Eim_batch)
    #print('min_Er_batch = ', min_Er_batch)   
    #print('min_Aim_batch = ', min_Aim_batch)
    #print('min_Eim_batch = ', min_Eim_batch)
 
    #max_batch = cp.max(cp.abs(cp.array([min_Er_batch, max_Er_batch, min_Ar_batch, max_Ar_batch, 
    #                                   min_Eim_batch, max_Eim_batch, min_Aim_batch, max_Aim_batch])))

    max_batch = cp.max(cp.array([max_Er_batch, max_Ar_batch, max_Eim_batch, max_Aim_batch]))
 
    print('max_batch = ', max_batch)

    #max_batch = max_batch + 100;
    #print('max_batch = ', max_batch)
    #maxE = cp.sqrt(max_Er_batch**2 + max_Eim_batch**2)/cp.sqrt(2.0*dt)

    #print('maxA = ',

    # Add frequencies as one of the channels
    #freqs = (gb.get_freqs() - config_data['limits']['min']['fvec'])/(config_data['limits']['max']['fvec'] - config_data['limits']['min']['fvec'])
    # TODO this has to be saved in the file in the case we are going to restart the training
    param_mean_cp = gb.get_param_mean()
    param_std_cp = gb.get_param_std()
    param_mean = torch.as_tensor(param_mean_cp).type(dtype)
    param_std = torch.as_tensor(param_std_cp).type(dtype)

    #Ar_mean_cp, Er_mean_cp, Aim_mean_cp, Eim_mean_cp = gb.get_wf_mean()
    #Ar_std_cp, Er_std_cp, Aim_std_cp, Eim_std_cp = gb.get_wf_std()

    #print('Ar_mean_cp = ', Ar_mean_cp)
    #print('Ar_std_cp = ', Ar_std_cp)

    #Ar_mean, Er_mean, Aim_mean, Eim_mean = torch.as_tensor(Ar_mean_cp).type(dtype)
    #Ar_std, Er_std, Aim_std, Eim_std = torch.as_tensor(param_std_cp).type(dtype)

    freqs = (gb.get_freqs() - param_mean_cp[0]) / param_std_cp[0]

    #freqs = (gb.get_freqs() - gb.get_freqs()[0])/(gb.get_freqs()[-1] - gb.get_freqs()[0])
    freqs_arr = torch.as_tensor(cp.tile(freqs,(batch_size, 1))).type(dtype).view(batch_size, 1, -1)
    freqs_one = torch.as_tensor(freqs).view(1,-1).type(dtype).view(1, 1, -1)
 
    # TEST WAVEFORMS
    test_wf = False
    if test_wf == True:
        index = 0
        lambd = config_data['default']['lam']
        beta = -config_data['default']['beta']
        A, E, truths = gb.check_waveforms(beta, lambd, index)
        waveform = torch.cat((torch.as_tensor((cp.real(A) - Ar_mean_cp)/Ar_std_cp).type(dtype),
                              torch.as_tensor((cp.real(E) - Er_mean_cp)/Er_std_cp).type(dtype),
                              torch.as_tensor((cp.imag(A) - Aim_mean_cp)/Aim_std_cp).type(dtype),
                              torch.as_tensor((cp.imag(E) - Eim_mean_cp)/Eim_std_cp).type(dtype))) #,  freqs_one), 1)
            
        waveform_cnn = torch.reshape(waveform, (waveform.shape[0], -1)) 
        #waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))
       
        flow.eval()
        with torch.no_grad():
         
            loss_val = -flow.log_prob(torch.as_tensor(truths).type(dtype).view(1,-1), waveform_cnn).mean()

   
    # EPOCHS
    for j0 in range(number_epochs):

        j = j0 + last_epoch + 1
        print('j = ', j)
 
        flow.train()
        for i in range(number_iterations):

            gb.sample_from_prior(batch_size, 1)
            A, E = gb.create_waveform(1)
         
            param = torch.as_tensor(gb.get_params()).type(dtype)
            #test = torch.as_tensor(cp.real(A)/max_batch).type(dtype).view(batch_size, 1, -1)     
    
            #print('param = ', param)
       
            waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype).view(batch_size, -1), 
                                   torch.as_tensor(cp.real(E)/max_batch).type(dtype).view(batch_size, -1), 
                                   torch.as_tensor(cp.imag(A)/max_batch).type(dtype).view(batch_size, -1), 
                                   torch.as_tensor(cp.imag(E)/max_batch).type(dtype).view(batch_size, -1)), 1)#, freqs_arr), 1)
            
            #waveform = torch.cat((torch.as_tensor((cp.real(A) - Ar_mean_cp)/Ar_std_cp).type(dtype).view(batch_size, -1),
            #                  torch.as_tensor((cp.real(E) - Er_mean_cp)/Er_std_cp).type(dtype).view(batch_size, -1),
            #                  torch.as_tensor((cp.imag(A) - Aim_mean_cp)/Aim_std_cp).type(dtype).view(batch_size, -1),
            #                  torch.as_tensor((cp.imag(E) - Eim_mean_cp)/Eim_std_cp).type(dtype).view(batch_size, -1)), 1)
            #print('torch.max(waveform) = ', torch.max(waveform))
            #print('torch.min(waveform) = ', torch.min(waveform)) 
    
            #print('cp.real(A) = ',  cp.real(A))

            test = cp.real(A).get()/max_batch.get()
            #print(test.shape)
            plt.figure()
            plt.plot(test[0,:])
            plt.plot(test[1,:])
            plt.savefig('wf.png')

            exit()
            #print((cp.real(A) - Ar_mean_cp)/Ar_std_cp)
            
          
            waveform_cnn = torch.reshape(waveform, (waveform.shape[0], -1))
            #waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))
   
            loss = -flow.log_prob(param, waveform_cnn).mean()
              
            #switch_optim(optimizer)

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
        checkpoint_path = config['saving']['save_root'] + 'checkpoint' + config['saving']['label'] + '_{}.pt'.format(str(j+1))
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
            gb.sample_from_prior(batch_size, 1)
            A, E = gb.create_waveform(1)
            param = torch.as_tensor(gb.get_params()).type(dtype)
            waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype).view(batch_size, -1), 
                                  torch.as_tensor(cp.real(E)/max_batch).type(dtype).view(batch_size, -1), 
                                  torch.as_tensor(cp.imag(A)/max_batch).type(dtype).view(batch_size, -1), 
                                  torch.as_tensor(cp.imag(E)/max_batch).type(dtype).view(batch_size, -1)),1)#, freqs_arr), 1)

            #waveform = torch.cat((torch.as_tensor((cp.real(A) - Ar_mean_cp)/Ar_std_cp).type(dtype).view(batch_size, -1),
            #                  torch.as_tensor((cp.real(E) - Er_mean_cp)/Er_std_cp).type(dtype).view(batch_size, -1),
            #                  torch.as_tensor((cp.imag(A) - Aim_mean_cp)/Aim_std_cp).type(dtype).view(batch_size, -1),
            #                  torch.as_tensor((cp.imag(E) - Eim_mean_cp)/Eim_std_cp).type(dtype).view(batch_size, -1)), 1)

            waveform_cnn = torch.reshape(waveform, (waveform.shape[0], -1))
            #waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))
          
            loss_val = -flow.log_prob(param, waveform_cnn).mean()

            print('loss_val = %.3f' % loss_val)
            losses_val.append(loss_val.tolist())

    
            # Do pp-plot and corner plot after each epoch to see how the distribution of the parameters is converging
            neval = 100    # number of injections
            num_samples = 1000
          
            # 'Real' data
            parameter_labels =  gb.get_param_label()
            #param_mean = torch.as_tensor(gb.get_param_mean()).type(dtype)
            #param_std = torch.as_tensor(gb.get_param_std()).type(dtype)

            A, E, truths = gb.true_data()

            waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype).view(1, -1), 
                                  torch.as_tensor(cp.real(E)/max_batch).type(dtype).view(1, -1), 
                                  torch.as_tensor(cp.imag(A)/max_batch).type(dtype).view(1, -1), 
                                  torch.as_tensor(cp.imag(E)/max_batch).type(dtype).view(1, -1)),1)#, freqs_one), 1)

            #waveform = torch.cat((torch.as_tensor((cp.real(A) - Ar_mean_cp)/Ar_std_cp).type(dtype).view(1, -1),
            #                  torch.as_tensor((cp.real(E) - Er_mean_cp)/Er_std_cp).type(dtype).view(1, -1),
            #                  torch.as_tensor((cp.imag(A) - Aim_mean_cp)/Aim_std_cp).type(dtype).view(1, -1),
            #                  torch.as_tensor((cp.imag(E) - Eim_mean_cp)/Eim_std_cp).type(dtype).view(1, -1)), 1)
 
           
            waveform_cnn = torch.reshape(waveform, (waveform.shape[0], -1))
            #waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))
        
            # Label for plots
            label = config['plots']['label'] 
            
            percentiles = np.empty((neval, features_size))
            print('Do corner plot')
            # TODO check what are the values of the truths and coeff_true
  
            filename_sample = config['samples']['path']
            amp_sample = config_data['default']['amp']
            freq_sample_min = config_data['limits']['min']['fsamp']
            freq_sample_max = config_data['limits']['max']['fsamp']   
            make_cp_compare_samples_gb(flow, j, parameter_labels, param_mean, param_std, waveform_cnn, truths, label, filename_sample, amp_sample, freq_sample_min, freq_sample_max)
                                     # flow, iteration, labels, param_mean, param_std, coeff_norm, truths, test_label, filename, amp_true, freq

            print('Do pp plot')
            for idx in tqdm(range(neval)):

                # This function has to pass a combination of A and E
                # Check if 'num_samples' works here correctly
                gb.sample_from_prior(1, 1)
                A, E = gb.create_waveform(1)
                param = torch.as_tensor(gb.get_params()).type(dtype)

                freqs_samp = torch.as_tensor(cp.tile(freqs,(num_samples, 1))).type(dtype).view(1, -1) 
                waveform = torch.cat((torch.as_tensor(cp.real(A)/max_batch).type(dtype).view(1, -1), 
                                      torch.as_tensor(cp.real(E)/max_batch).type(dtype).view(1, -1), 
                                      torch.as_tensor(cp.imag(A)/max_batch).type(dtype).view(1, -1), 
                                      torch.as_tensor(cp.imag(E)/max_batch).type(dtype).view(1,  -1)),1)#, freqs_one), 1)

#                waveform = torch.cat((torch.as_tensor((cp.real(A) - Ar_mean_cp)/Ar_std_cp).type(dtype).view(1, -1),
#                              torch.as_tensor((cp.real(E) - Er_mean_cp)/Er_std_cp).type(dtype).view(1, -1),
#                              torch.as_tensor((cp.imag(A) - Aim_mean_cp)/Aim_std_cp).type(dtype).view(1, -1),
#                              torch.as_tensor((cp.imag(E) - Eim_mean_cp)/Eim_std_cp).type(dtype).view(1, -1)), 1)
 

                waveform_cnn = torch.reshape(waveform, (waveform.shape[0], -1))
                #waveform_cnn = torch.reshape(waveform, (waveform.shape[0], embed_in, 1, -1))
         
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

