'''

  Check if SVD decomposition is good enough.
  Use iterative PCA to add more elements.

'''
import torch
import torch.nn as nn
import sys
import argparse
#sys.path.append('../mbhbs/')
from data_generation.mbhbs.mbhb_model_Lframe import MBHB_gpu
#from mbhb_model_Lframe import MBHB_gpu

from flow.utils.torchutils import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import offsetbox
import h5py



# Calculate the overlap between two waveforms
def overlap(wf, wf_recover):

    return torch.dot(wf, wf_recover)/(torch.sqrt(torch.dot(wf, wf))*torch.sqrt(torch.dot(wf_recover, wf_recover)))


def dot_prod(wf1, wf2):

    return torch.bmm(wf1.unsqueeze(dim=1), wf2.unsqueeze(dim=2)).squeeze()

# Calculate the overlap between two waveforms (use cupy tensor)
def overlap_tensor(wf1, wf2):

    return dot_prod(wf1, wf2)/(torch.sqrt(dot_prod(wf1, wf1))*torch.sqrt(dot_prod(wf2, wf2)))

 

# Center and scale waveforms by the variance of an ansamble for each point in the time series
def scale(x, mean, std):

    return (x - mean)/std


def train_AE(mbhb, dev, dtype):

    # Create one samle of the data to check the size of the dataset
    mbhb.freqwave_AET(10000)
    mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
    Nt = mbhb_ts.shape[1]
    mean_batch = torch.mean(mbhb_ts, dim=0)
    std_batch = torch.std(mbhb_ts, dim=0)
  
    #mbhb.freqwave_AET(10)
    #mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)

    #norm_test = scale(mbhb_ts, mean_batch, std_batch)
    
    #plt.figure()
    #plt.plot(norm_test[0].detach().cpu().numpy())
    #plt.savefig('norm_test.png')

    
    model = AE(Nt).to(dev)
    criterion = nn.MSELoss()

    learning_rate = 2e-4

    num_epochs = 100000
    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30000, eta_min=0, last_epoch=-1)

    for epoch in range(num_epochs):
        # input data with the noise, output data without the noise (will it not corrupt the noise estimation?)
        # otherwise we can add the white noise to the reduced representation of the data

        Nsamples = 1024
        mbhb.freqwave_AET(Nsamples)
        mbhb_ts = mbhb.timewave_AET() 

        #mbhb.freqwave_AET(batch_size)
        mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
        mbhb_ts_scale = scale(mbhb_ts, mean_batch, std_batch)

        noise_samples = torch.randn(mbhb_ts.shape).type(dtype)

        mbhb_ts_noise = mbhb_ts #+ noise_samples        
        mbhb_ts_noise_scale = scale(mbhb_ts_noise, mean_batch, std_batch)

        model.train()
        output = model(mbhb_ts_noise_scale)
        loss = criterion(output, mbhb_ts_scale)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
   
        # Check overlap for reconstructed waveform
        if epoch % 100 == 0:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
            model.eval()

            mbhb.freqwave_AET(1)
            mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
            mbhb_ts_scale = scale(mbhb_ts, mean_batch, std_batch)

            noise_samples = torch.randn(mbhb_ts.shape).type(dtype)

            mbhb_test = mbhb_ts #+ noise_samples  # samples from the waveform to check the reconstruction
            mbhb_test_scale = scale(mbhb_test, mean_batch, std_batch)

            output_test = model(mbhb_test_scale)
            loss = criterion(output_test, mbhb_ts_scale) 
            print('loss = %.3f' % loss)
            print('overlap = ', overlap(mbhb_ts_scale[0,:], output_test[0,:]))
            for param_group in optimizer.param_groups:
                print( param_group['lr'])

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../../configs/mbhbs/mbhb_resample_all.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='../../configs/mbhbs/mbhb_data_radler_no_time_dist.yaml',
                        help='Path to config file specifying parameters of the source when we sample on the fly')
  
    args = parser.parse_args()

    # Load config
    config = get_config(args.config)
    config_data = get_config(args.config_data)


    # Choose CPU or GPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        dtype = torch.cuda.FloatTensor
    else:
        dev = "cpu"
        dtype = torch.FloatTensor
    print('device = ', dev)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
   
    # Initialise the class for MBHB waveforms
    mbhb = MBHB_gpu(config, config_data, dtype)
    mbhb.freqwave_AET(1000)
    mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
    
    num_coeff = 128

    fin = h5py.File('../../' + config['svd']['root'] + config['svd']['path'], 'r')
    Sts = fin['S']
    Vts = fin['V']

    # Take only first elements of the coefficients
    Vts_reduce = torch.from_numpy(Vts[:,:num_coeff]).type(dtype)
    #Vt = torch.from_numpy(np.transpose(Vts_reduce)).type(dtype)
    Sts_reduce = torch.from_numpy(Sts[:num_coeff]).type(dtype)

    # Generate new data to test reconstruction
    Nsamples = 100
    mbhb.freqwave_AET(Nsamples)
    mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
    
    coeff = torch.matmul(mbhb_ts, Vts_reduce)
    coeff_scale = (coeff)/torch.sqrt(Sts_reduce)  
    
    project_back = torch.sqrt(Sts_reduce)*Vts_reduce   
    project_back_t = torch.transpose(project_back,0,1)
    x_wf_recover = torch.matmul(coeff_scale, project_back_t)

    print('overlap = ', overlap_tensor(mbhb_ts, x_wf_recover))





if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)


















