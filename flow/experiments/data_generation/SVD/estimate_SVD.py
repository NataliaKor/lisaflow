'''

  Check if SVD decomposition is good enough.
  Use iterative PCA to add more elements.

'''
import torch
import torch.nn as nn
import sys
import argparse
#sys.path.append('../mbhbs/')
from flow.experiments.data_generation.mbhbs.mbhb_model_vary_time import MBHB_gpu
#from mbhb_model_Lframe import MBHB_gpu

from flow.utils.torchutils import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import offsetbox
import h5py

#from sklearn.decomposition import IncrementalPCA

from cuml.decomposition import IncrementalPCA

import cupy as cp

import cupyx


# Use cupy accelerated implementation of sklearn incremental PCA.
def iPCA(mbhb, dtype):

    num_iterations = 100
 
    ipca = IncrementalPCA(n_components=150, batch_size = 4096)
 
    for i in range(num_iterations):
       mbhb.sample_from_prior(4096,0)
       mbhb_ts = mbhb.create_waveform()
       ipca.partial_fit(mbhb_ts) 
       #print(ipca.components_) 
       print('Singular values:')
       print(ipca.singular_values_)
       print('S.shape = ', ipca.singular_values.shape)
       print('V.shape = ', )
       print('Variance ratios:')
       print(ipca.explained_variance_ratio_)
       print('Cumsum:')
       print(cp.cumsum(ipca.explained_variance_ratio_))
       print(' ')
       if i % 1 == 0:
           mbhb.sample_from_prior(16, 1)
           mbhb_ts = torch.as_tensor(mbhb.create_waveform()).type(dtype)
           mbhb_ts_reconstruct = torch.as_tensor(ipca.inverse_transform(ipca.transform(mbhb_ts))).type(dtype)
           loss = overlap_tensor(mbhb_ts, mbhb_ts_reconstruct)  
          
           print('loss.shape = ', loss.shape)             
           print('overlap = ', loss)
           print('i [{}/{}]'.format(i + 1, num_iterations))

       print('Save results of te fit to the file')
       #f = h5py.File(config['svd']['root'] + config['svd']['path'], 'w')
       #f.create_dataset("S", data=Sts[:num_coeff].detach().cpu().numpy())
       #f.create_dataset("V", data=Vts[:,:num_coeff].detach().cpu().numpy())

# Simple autoencoder to see if we can compress the data well enough  
class AE(nn.Module):
    def __init__(self, Nt):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(Nt, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, Nt),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
    parser.add_argument('--config', type=str, default='../../configs/mbhbs/mbhb_resample_256.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='../../configs/mbhbs/mbhb_data_no_dist.yaml',
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
    #mbhb.freqwave_AET(4048)
    #mbhb_ts = mbhb.timewave_AET()

    iPCA(mbhb, dtype)
 
    exit()

    train_AE(mbhb, dev, dtype)
    exit()

    # Initialise a class for MBHB
    Nsamples = 17000#25000
    mbhb.freqwave_AET(Nsamples)
    mbhb_ts = mbhb.timewave_AET()

    test = torch.as_tensor(mbhb_ts).to_sparse()

    Uts, Sts, Vts = torch.svd_lowrank(test, q = 32) # torch.svd(torch.as_tensor(mbhb_ts))

    f = h5py.File('testSVD.hdf5', 'w')
    f.create_dataset("S", data=Sts.detach().cpu().numpy())
    f.create_dataset("U", data=Uts.detach().cpu().numpy())
    f.create_dataset("V", data=Vts.detach().cpu().numpy())

    num_coeff = 32

    # Take only first elements of the coefficients
    # TODO maybe write to file only the reduced arrays to save space
    Vts_reduce = Vts[:,:num_coeff].type(dtype)
    Sts_reduce = Sts[:num_coeff].type(dtype)

    # Generate new data to test reconstruction
    Nsamples = 100
    mbhb.freqwave_AET(Nsamples)
    mbhb_ts = torch.as_tensor(mbhb.timewave_AET()).type(dtype)
    
    coeff = torch.matmul(mbhb_ts, Vts_reduce)
    coeff_scale = (coeff)/torch.sqrt(Sts_reduce)  
    print('coeff_scale = ', coeff_scale)
 
    project_back = torch.sqrt(Sts_reduce)*Vts_reduce
    print('project_back.shape = ', project_back.shape)
    project_back_t = torch.transpose(project_back,0,1)


    x_wf_recover = torch.matmul(coeff, project_back_t)

    overlap = torch.dot(mbhb_ts[0], x_wf_recover[0])/(torch.sqrt(torch.dot(mbhb_ts[0], mbhb_ts[0]))*torch.sqrt(torch.dot(x_wf_recover[0],x_wf_recover[0])))
    print('overpal = ', overlap)


    ipca = IncrementalPCA(n_components = num_coeff, batch_size = 1024) #8192)

    for i in range(10):
        print('i = ', i)
        mbhb.freqwave_AET(1024)
        mbhb_ts = mbhb.timewave_AET()
        ipca.partial_fit(mbhb_ts.get())






if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train flow for the simple example chirplet')
    #args = parser.parse_args()
    main(parser)


















