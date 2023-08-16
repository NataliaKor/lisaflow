"""
Train the network with the simple chirplet model.
Create data on the fly.
"""
import argparse
import numpy as np
from tqdm import tqdm
from scipy import stats
import torch
from torch.nn.utils import clip_grad_norm_
from flow.utils.monitor_progress import *
from flow.utils.torchutils import *
from flow.distributions.normal import *
from flow.distributions.resample import *
from flow.networks.mlp import MLP
from flow_architecture import *
from data_generation.toy.chirp_model import chirplet_normalised

def sample_chirp_norm(nsamples, config_data, tvec, Sts_reduce, Vts_reduce, dtype, with_noise):

    '''
      Sample training data on the fly.
    '''
    # Define values for fixed parameters
    # and boundaries for varied parameters
    parameter_labels = []
    param_min = []
    param_max = []
    if config_data['estimate']['Q']:
        Q_min = config_data['limits']['min']['Q']
        Q_max = config_data['limits']['max']['Q']
        parameter_labels.append('Q')
        param_min.append(Q_min)
        param_max.append(Q_max)
        Q_rand = np.random.uniform(low = Q_min, high = Q_max, size = nsamples)
    else: 
        Q = config_data['default']['Q']

    if config_data['estimate']['t0']:
        t0_min = config_data['limits']['min']['t0']
        t0_max = config_data['limits']['max']['t0']
        parameter_labels.append('t0')
        param_min.append(t0_min)
        param_max.append(t0_max)
        t0_rand = np.random.uniform(low = t0_min, high = t0_max, size = nsamples)
    else:
        t0 = config_data['default']['t0']

    if config_data['estimate']['f0']:
        f0_min = config_data['limits']['min']['f0']
        f0_max = config_data['limits']['max']['f0']
        parameter_labels.append('f0')
        param_min.append(f0_min)
        param_max.append(f0_max)
        f0_rand = np.random.uniform(low = f0_min, high = f0_max, size = nsamples)
    else:
        f0 = config_data['default']['f0']

    if config_data['estimate']['fdot']:
        fdot_min = config_data['limits']['min']['fdot']
        fdot_max = config_data['limits']['max']['fdot']
        parameter_labels.append('fdot')
        param_min.append(fdot_min)
        param_max.append(fdot_max)
        fdot_rand = np.random.uniform(low = fdot_min, high = fdot_max, size = nsamples)
    else:
        fdot = config_data['default']['fdot']

    #Create arrays of max and min values that we beed to use in the future to normalise the data back to the correct values
    param_min_arr = np.array(param_min)
    param_max_arr = np.array(param_max)

    chirp_rand = np.zeros((nsamples, tvec.shape[0]))
    param = torch.Tensor().type(dtype)

    for j in range(nsamples):
        if config_data['estimate']['Q']:
            Q = Q_rand[j]
        if config_data['estimate']['t0']:
            t0 = t0_rand[j]
        if config_data['estimate']['f0']:
            f0 = f0_rand[j]
        if config_data['estimate']['fdot']:
            fdot = fdot_rand[j]

        chirp_rand[j,:] = chirplet_normalised(tvec, Q, t0, f0, fdot)

    if config_data['estimate']['Q']:
        Q_norm = torch.from_numpy(normalise(Q_rand, Q_min, Q_max)).type(dtype).view(-1,1)
        param = torch.cat([param, Q_norm], dim=1)
    if config_data['estimate']['t0']:
        t0_norm = torch.from_numpy(normalise(t0_rand, t0_min, t0_max)).type(dtype).view(-1,1)
        param = torch.cat([param, t0_norm], dim=1)
    if config_data['estimate']['f0']:
        f0_norm = torch.from_numpy(normalise(f0_rand, f0_min, f0_max)).type(dtype).view(-1,1)
        param = torch.cat([param, f0_norm], dim=1)
    if config_data['estimate']['fdot']:
        fdot_norm = torch.from_numpy(normalise(fdot_rand, fdot_min, fdot_max)).type(dtype).view(-1,1)
        param = torch.cat([param, fdot_norm], dim=1)
 
    chirp_wf = torch.from_numpy(chirp_rand).type(dtype)

    if with_noise:
        noise_samples = 0.1*torch.randn(chirp_wf.shape).type(dtype)
        coeff = torch.matmul(chirp_wf + noise_samples, torch.from_numpy((1.0/np.sqrt(Sts_reduce + 1))*Vts_reduce).type(dtype))
    else:
        coeff = torch.matmul(chirp_wf, torch.from_numpy((1.0/np.sqrt(Sts_reduce))*Vts_reduce).type(dtype))

    return coeff, param, param_min_arr, param_max_arr, parameter_labels


def true_data(config_data, t, Sts_reduce, Vts_reduce, dtype, with_noise):
    '''
      "True" data for testing. Dataset with the default parameters.
    '''
   
    Q  = config_data['default']['Q']
    t0 = config_data['default']['t0']

    f0 = config_data['default']['f0']
    fdot = config_data['default']['fdot']

    param = []
    if config_data['estimate']['Q']:
        param.append(Q)
    if config_data['estimate']['t0']:
        param.append(t0)
    if config_data['estimate']['f0']:
        param.append(f0)
    if config_data['estimate']['fdot']:
        param.append(fdot)

    x_wf = torch.from_numpy(chirplet_normalised(t, Q, t0, f0, fdot)).type(dtype)

    if with_noise:
        noise_samples = 0.1*torch.randn(x_wf.shape).type(dtype)
        coeff = torch.matmul(x_wf + noise_samples, torch.from_numpy((1.0/np.sqrt(Sts_reduce + 1))*Vts_reduce).type(dtype)).view(1,-1)
    else:
        coeff = torch.matmul(x_wf, torch.from_numpy((1.0/np.sqrt(Sts_reduce))*Vts_reduce).type(dtype)).view(1,-1)

    #coeff = torch.matmul(x_wf, torch.from_numpy((1.0/np.sqrt(Sts_reduce))*Vts_reduce).type(dtype)).view(1,-1)

    return coeff, param



# Normalise parameters to be from 0 to 1
def normalise(par, par_min, par_max):

    return (par - par_min)/(par_max - par_min)


def normalise_coeff(coeff, coeff_min, coeff_max):
    return -1.0 + 2.0*(coeff - coeff_min)/(coeff_max - coeff_min)

# Scale all coefficients
def scale_coeff(coeff, coeff_maxabs):
  
    return coeff/coeff_maxabs


def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='configs/toy/chirp_resample.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='configs/toy/chirp_data_onthefly.yaml',
                        help='Path to config file specifying parameters of the source when we sample on the fly')
    parser.add_argument('--resume', action='store_true', help='Flag whether to resume training')

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

    # Create data on the fly:
    # Define time vector
    fs = config_data['tvec']['fs']
    dt = 1.0/fs
    T =  config_data['tvec']['T']
    t = np.arange(0, T, dt)    

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
        in_shape = [2],
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
    # Set optimisers and schedulers

    # Choose optimiser 
    optimizer = optim.Adam(flow.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
 
    # TODO save scheduler in the checkpoint and load it
    # Choose to resume training from the previous training results or start fresh
    if config['training']['resume']:
        checkpoint = torch.load(config['saving']['save_root'] + config['training']['checkpoints'])
        flow.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        learning_rate = 1.7e-4
        #for g in optimizer.param_groups:
        #    g['lr'] = learning_rate 
    else:
        last_epoch = -1

   # Schedule for learning rate annealing
    if anneal_learning_rate:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['training']['num_training_steps'], eta_min=0, last_epoch=last_epoch)
    else:
        scheduler = None

    ##############################################################################################################################
    # Load SVD decomposition

    # Load the results of the SVD
    fin = h5py.File(config['svd']['root'] + config['svd']['path'], 'r')
    Sts = fin['Sts']
    Vts = fin['Vts']

    # Take only first elements of the coefficients
    Vts_reduce = Vts[:,:num_coeff]
    Vt = np.transpose(Vts_reduce)
    Sts_reduce = Sts[:num_coeff]

    # Estimate min and max values for big batch of samples to normalise it
    coeff, param, _, _, _ = sample_chirp_norm(10000, config_data, t, Sts_reduce, Vts_reduce, dtype, 0)
    coeff_max  = torch.max(coeff)
    coeff_min = torch.min(coeff)
    # Check which absolute value is larger for scaling
    if torch.abs(coeff_max) >= torch.abs(coeff_min):
        coeff_maxabs = coeff_max
    else:
        coeff_maxabs = torch.abs(coeff_min)
    print('coeff_maxabs = ', coeff_maxabs) 

    # EPOCHS
    for j0 in range(number_epochs):

        j = j0 + last_epoch + 1
        print('j = ', j)
 
        flow.train()
        for i in range(number_iterations):

            coeff, param, _, _, _ = sample_chirp_norm(batch_size, config_data, t, Sts_reduce, Vts_reduce, dtype, 1)
            #coeff_norm = scale_coeff(coeff, coeff_maxabs)

            loss = -flow.log_prob(param, coeff).mean()
           
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
           'loss': loss,}, checkpoint_path)
        np.savetxt(config['saving']['save_root'] + 'losses_' + config['saving']['label'] + '.txt', losses)
        np.savetxt(config['saving']['save_root']+ 'losses_val' + config['saving']['label'] + '.txt', losses_val)

        # Evaluate, and record corner and pp plots
        flow.eval()
        with torch.no_grad():

            coeff, param, param_min_arr, param_max_arr, parameter_labels = sample_chirp_norm(batch_size, config_data, t, Sts_reduce, Vts_reduce, dtype, 1)
            #coeff_norm = scale_coeff(coeff, coeff_maxabs)
 
            loss_val = -flow.log_prob(param, coeff).mean()

            print('loss_val = %.3f' % loss_val)
            losses_val.append(loss_val.tolist())

    
            # Do pp-plot and corner plot after each epoch to see how the distribution of the parameters is converging
            neval = 100    # number of injections
            num_samples = 5000
          
            # 'Real' data
            coeff_true, truths = true_data(config_data, t, Sts_reduce, Vts_reduce, dtype, 1)
            coeff_true_norm = scale_coeff(coeff_true, coeff_maxabs)

            # Label for plots
            label = config['plots']['label'] 

            percentiles = np.empty((neval, features_size))
            print('Start doing pp plot')
            for idx in tqdm(range(neval)):

                coeff, param, _, _, _ = sample_chirp_norm(1, config_data, t, Sts_reduce, Vts_reduce, dtype, 1)
                #coeff_norm = scale_coeff(coeff, coeff_maxabs)

                samples = flow.sample(num_samples, coeff).squeeze().cpu().detach().numpy()
                parameters_true = param.cpu().detach().numpy()

                for n in range(features_size):
                    percentiles[idx, n] = stats.percentileofscore(samples[:,n], parameters_true[0,n])

            make_pp(percentiles, parameter_labels, j, label)
            make_cp(flow, j, parameter_labels, param_min_arr, param_max_arr, coeff_true_norm, truths, label)
            gc.collect()



if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)

