"""
Train the network with the simple chirplet model.
Create data on the fly.
"""
#from data_generation.mbhbs.mbhb_model_std_norm import MBHB_gpu
from data_generation.mbhbs.mbhb_model_save_data import MBHB_gpu

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

#from flow.utils.adahessian2 import AdaHessian
#from flow.utils.stepLR import stepLR
from flow.utils.adahessian import Adahessian, get_params_grad

from flow.distributions.normal import *
from flow.distributions.resample import *
from flow.networks.mlp import MLP

from flow_architecture_play import *

from cuml.decomposition import IncrementalPCA


#from data_generation.mbhbs.mbhb_model import MBHB_gpu
#from data_generation.mbhbs.mbhb_model_Lframe import MBHB_gpu

# From SWATS https://github.com/Mrpatekful/swats
def switch_optim(optimizer):
    switch = 0
    for group in optimizer.param_groups:
        for w in group['params']:
            if w.grad is None:
                continue
            grad = w.grad.data

            beta1, beta2 = group['betas']
            step = optimizer.state[optimizer.param_groups[0]["params"][-1]]["step"]
            #__getstate__()

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
                #if state['step'] > 1 and corrected_exp_avg.allclose(scaling, rtol=1e-6) and corrected_exp_avg > 0:
                if step > 1 and corrected_exp_avg[0].isclose(scaling, atol=5e-5) and corrected_exp_avg > 0: # rtol = 1e-6
                  
                    print('Switch to SGD')
                    #group['phase'] = 'SGD'
                    print('lr = ', corrected_exp_avg.item())
                    #group['lr'] = corrected_exp_avg.item()
                    switch = 1     
    return switch    


def normalise_coeff(coeff, coeff_min, coeff_max):

    return -1.0 + 2.0*(coeff - coeff_min)/(coeff_max - coeff_min)

# Scale all coefficients

  
#    return coeff/coeff_maxabs


def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/mbhbs/mbhb_resample_256.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='configs/mbhbs/mbhb_data_no_dist.yaml',
                        help='Path to config file specifying parameters of the source when we sample on the fly')
    parser.add_argument('--resume', type=int, default=1, help='Flag whether to resume training')

    args = parser.parse_args() 

    # Choose CPU or GPU
    if torch.cuda.is_available():
        dev = "cuda:1"
        dev1 = "cuda:0"
        dtype = torch.cuda.FloatTensor
    else:
        dev = "cpu"
        dtype = torch.FloatTensor
    print('device = ', dev)
    print('device1 = ', dev1)
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
    number_iterations =  config['training']['max_iter']
    grad_norm_clip_value = config['training']['grad_norm_clip_value']
    anneal_learning_rate =  config['training']['anneal_learning_rate']

    # Record losses
    losses = []
    losses_val = []
 
    # Define number of coefficients and context size after embedding
    num_coeff = config['model']['context']['coeffs']
    context_features_size = config['model']['context']['context_features']
   
    # Cadence
    dt = config_data['tvec']['dt']
 
    # Size of the physical parameters
    features_size = config['model']['base']['params']

    # Define base distribution. At the moment there are 2 options: 
    if config['model']['base']['gaussian']:
        distribution = StandardNormal((features_size,)).to(dev1)
    else:
        acceptance_fn = MLP(
        in_shape = [features_size],
        out_shape = [1],
        hidden_sizes = [512, 512],
        activation = F.leaky_relu,
        activate_output = True,
        activation_output = torch.sigmoid)
        distribution = ResampledGaussian((features_size,), acceptance_fn).to(dev1) 
       
    transform = create_transform(config).to(dev1)

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
            use_batch_norm = config['model']['embedding']['batch_norm']).to(dev1)

    flow = Flow(transform, distribution, embedding_net).to(dev1)    
    #flow = Flow(transform, distribution).to(dev)
    #########################################################################################################################
    # Set optimisers and schedulers
  
    # Choose optimiser
    optim_method = config['training']['optimizer']
    if optim_method == 'SDG':
        optimizer = optim.SGD(flow.parameters(), lr=config['training']['learning_rate'])
        optim_type = 1
    elif optim_method == 'Adam':
        optimizer = optim.Adam(flow.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        optim_type = 0
    elif optim_method == 'Hessian':
        optimizer = Adahessian(flow.parameters(), lr=config['training']['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=config['training']['weight_decay']) 
        #optimizer = Adahessian(flow.parameters(), lr=config['training']['learning_rate'])
        #optimizer = optim.RAdam(flow.parameters(), lr=config['training']['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=config['training']['weight_decay'])
        optim_type = 1
    # Schedule for learning rate annealing
    if anneal_learning_rate:
        if optim_method == 'Hessian':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[30,45],gamma=.1,last_epoch=-1)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['training']['num_training_steps'], eta_min=0, last_epoch=-1) # -1
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

        if config['training']['optimizer'] == 'SDG':
            optimizer = optim.SGD(flow.parameters(), lr=config['training']['learning_rate'])
            if anneal_learning_rate:
                #scheduler = lr_scheduler.MultiStepLR(optimizer,[30,45],gamma=.1,last_epoch=-1)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['training']['num_training_steps'], eta_min=0, last_epoch=-1) # -1
            else:
                scheduler = None
            # Dummy loop to advance the learning rate
            # Make the current iteration a variable
            if anneal_learning_rate:
                for _ in range(76):
                    scheduler.step()
 
    else:
        last_epoch = -1

       

    #############################################################################################################################
 
    # Calculate decomposition if it is the first run.
    # Load decomposition if it is subsecuent run with the same setup.
   
    ##############################################################################################################################
   
    # Initialise the class for MBHB waveforms
    # Here we have to specify that this has to be calculated on the second GPU
    mbhb = MBHB_gpu(config, config_data, dtype)
   
    batch_pca = 256

    # Use iPCA to estimate principle components
    if config['svd']['estimate']:
        if config['svd']['method'] == 'ipca':

            mbhb.sample_from_prior(256, 0)
            print('Generated initial sample')
            num_iterations = 5
            ipca = IncrementalPCA(n_components=128, batch_size = batch_pca)    
            # Here we have to run generation for the first time
            # because we have to fix mean and variance            
            for i in range(num_iterations):
                mbhb.sample_from_prior(batch_pca, 1)
                mbhb_ts = mbhb.create_waveform()
                ipca.partial_fit(mbhb_ts)
                print('i = ', i)
                print('Cumsum:')
                print(cp.cumsum(ipca.explained_variance_ratio_)) 
                print('S.shape = ', ipca.singular_values_.shape)
                print('V.shape = ', ipca.components_.shape)

            Vts = cp.transpose(ipca.components_).get()
            Sts = ipca.singular_values_.get()

        else:
      
            # Initialise a class for MBHB
            Nsamples = 10000
            mbhb.sample_from_prior(Nsamples, 0)
            mbhb_ts = mbhb.create_waveform()
            Uts_full, Sts_full, Vts_full = torch.svd(torch.as_tensor(mbhb_ts))
            Vts = Vts_full[:,:num_coeff].type(dtype).detach().cpu().numpy()
            Sts = Sts_full[:num_coeff].type(dtype).detach().cpu().numpy()   

        f = h5py.File(config['svd']['root'] + config['svd']['path'], 'w')
        f.create_dataset("S", data=Sts)    
        f.create_dataset("V", data=Vts) 

        # Take only first elements of the coefficients
        print('Vts_reduce.shape = ', Vts.shape)
        print('Sts_reduce.shape = ', Sts.shape)
        print('type(Vts_reduce) = ', type(Vts))
    
    else:
        fin = h5py.File(config['svd']['root'] + config['svd']['path'], 'r')
        Sts = fin['S']
        Vts = fin['V']

        # Take only first elements of the coefficients
        #Vts_reduce = torch.from_numpy(Vts[:,:num_coeff]).type(dtype)
        #Vt = torch.from_numpy(np.transpose(Vts_reduce)).type(dtype)
        #Sts_reduce = torch.from_numpy(Sts[:num_coeff]).type(dtype)

    Vts_reduce = torch.from_numpy(Vts[:,:]).type(dtype)
    Sts_reduce = torch.from_numpy(Sts[:]).type(dtype)


   
    # Do the first run to estimate the mean and variance for a set of parameters to normalise them properly.
    # This also has to be done on the second GPU.
    #Nsamples = 10000
    #mbhb.sample_from_prior(Nsamples, 0)

    #mbhb.sample_from_prior(1, 1) 
    #mbhb.create_waveform()
    #wf_true_cp, truths = mbhb.true_data()
  
    ###########################################################################################
    # Not enough memory on the GPU to create the data with the good resolution
    # Make two loops, in the first one create the data and record it on the hard drive.
    # When enough data is created do the pass of the gradient adjustment
    ############################################################################################
    # EPOCHS
    for j0 in range(number_epochs):

        j = j0 + last_epoch + 1
        print('j = ', j)
 
        flow.train()
        for i in range(number_iterations):

            # Record one batch of the data to the file
            output_file_time = 'MBHBs_time.hdf5' 
            with h5py.File(output_file_time, 'w') as f:
            
                # Here we have to pass device1 to make sure that all the data is created on the device1
                mbhb.sample_from_prior(batch_size, 1)
                mbhb_ts = torch.as_tensor(mbhb.create_waveform()).type(dtype)

                noise_samples = torch.randn(mbhb_ts.shape).type(dtype)
           
                coeff = torch.matmul(mbhb_ts + noise_samples, Vts_reduce)
                coeff_scale = coeff/Sts_reduce

                # Return parameters that correspond to the chosen coefficients
                #param = torch.as_tensor(mbhb.get_params()).type(dtype)
                param = mbhb.get_params()
                # 'mu', 'q', 'a1', 'a2', 'inc_cos', 'lamL', 'betaL_sin', 'psiL', 'phi_ref', 'offset'

                print('params.shape = ', param.shape)
                d_coeffs = f.create_dataset('coeffs', data = coeff_scale.detach().cpu().numpy())
                d_params = f.create_dataset('params', data = param.get())

             
            #mbhb.sample_from_prior(batch_size, 1)
            #mbhb_ts = torch.as_tensor(mbhb.create_waveform()).type(dtype)
           
            ##plt.figure()
            ##plt.plot(mbhb_ts[0,:].cpu().detach().numpy())
            ##plt.savefig('mbhb_test0.png')
            
            #noise_samples = torch.randn(mbhb_ts.shape).type(dtype)
            ##coeff = torch.matmul(mbhb_ts + noise_samples, (1.0/torch.sqrt(Sts_reduce + 1.0))*Vts_reduce)
         
            #coeff = torch.matmul(mbhb_ts + noise_samples, Vts_reduce) 
            #coeff_scale = coeff/Sts_reduce 

            #plt.figure()
            #plt.plot((mbhb_ts[0,:] + noise_samples[0,:]).cpu().detach().numpy())
            #plt.savefig('mbhb_noise_test0.png')
 
            
            # Return parameters that correspond to the chosen coefficients
            #param = torch.as_tensor(mbhb.get_params()).type(dtype)
   
            f_data = h5py.File('MBHBs_time.hdf5', 'r')
            coeff_scale = torch.from_numpy(f_data['coeffs']).type(dtype)
            param = torch.from_numpy(f_data['params']).type(dtype)
            loss = -flow.log_prob(param, coeff_scale).mean()

            if optim_type == 0:
                switch = switch_optim(optimizer)
                if switch == 1:
                    print('we are switching')
                    for param_group in optimizer.param_groups:
                        new_lr = param_group['lr']
                    optimizer = optim.SGD(flow.parameters(), lr=new_lr, weight_decay=config['training']['weight_decay'])
                    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['training']['num_training_steps'], eta_min=0, last_epoch=-1)
                    optim_type = 1 


            optimizer.zero_grad()
            if optim_method == 'Hessian':
                loss.backward(create_graph=True)
                _, gradsH = get_params_grad(flow)
               
            else:
                loss.backward(retain_graph=True)
            if grad_norm_clip_value > 0:
                clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
            if optim_method == 'Hessian':
                optimizer.step(gradsH)
            else:
                optimizer.step()

            # Check if we need to switch the optimiser
            #switch_optim() 

            #if anneal_learning_rate:
            #    scheduler.step()
            print('i = ', i, 'loss = %.3f' % loss)
            # Output current value of the learning rate
            for param_group in optimizer.param_groups:
                print( param_group['lr'])
            
            gc.collect()

        print('optim_type = ', optim_type)
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

        flow.eval()
        with torch.no_grad():

            # Generate waveforms and parameters
            mbhb.sample_from_prior(batch_size, 1)
            mbhb_ts = torch.as_tensor(mbhb.create_waveform()).type(dtype)
            noise_samples = torch.randn(mbhb_ts.shape).type(dtype) 
            coeff = torch.matmul(mbhb_ts + noise_samples, Vts_reduce)  
            coeff_scale = coeff/Sts_reduce
            param = torch.as_tensor(mbhb.get_params()).type(dtype)

            loss_val = -flow.log_prob(param, coeff_scale).mean()

            print('loss_val = %.3f' % loss_val)
            losses_val.append(loss_val.tolist())

  
            # Do pp-plot and corner plot after each epoch to see how the distribution of the parameters is converging
            neval = 100    # number of injections
            num_samples = 5000
          
            # 'Real' data
            parameter_labels =  mbhb.get_param_label()
        
            wf_true_cp, truths = mbhb.true_data()
            wf_true = torch.as_tensor(wf_true_cp).type(dtype)
            noise_samples = torch.randn(wf_true.shape).type(dtype)
 
            coeff_true = torch.matmul(wf_true + noise_samples, Vts_reduce)
            #noise_samples = torch.randn(coeff_true.shape).type(dtype)
            coeff_scale_true = coeff_true/Sts_reduce
        
            #    coeff = torch.matmul(x_wf, torch.from_numpy((1.0/np.sqrt(Sts_reduce))*Vts_reduce).type(dtype)).view(1,-1)

            #coeff_true_norm = scale_coeff(coeff_true, coeff_maxabs)

            # Label for plots
            label = config['plots']['label'] 
            param_mean = torch.as_tensor(mbhb.get_param_mean()).type(dtype)
            param_std = torch.as_tensor(mbhb.get_param_std()).type(dtype)


            percentiles = np.empty((neval, features_size))
            print('Do corner plot')
            # TODO check what are the values of the truths and coeff_true
            # make_cp(flow, j, parameter_labels, np.array(param_min), param_max, coeff_scale_true, truths, label)
            make_cp_as_std(flow, j, parameter_labels, param_mean, param_std, coeff_scale_true, truths, label)

            torch.cuda.empty_cache() 
            print('Do pp plot')
            for idx in range(neval): #tqdm(range(neval)):
                mbhb.sample_from_prior(1, 1)
                mbhb_ts = torch.as_tensor(mbhb.create_waveform()).type(dtype)
                #mbhb.freqwave_AET(1)
                #mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
                noise_samples = torch.randn(mbhb_ts.shape).type(dtype)
                coeff = torch.matmul(mbhb_ts + noise_samples, Vts_reduce)
                coeff_scale = coeff/Sts_reduce
                param = torch.as_tensor(mbhb.get_params()).type(dtype)

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

