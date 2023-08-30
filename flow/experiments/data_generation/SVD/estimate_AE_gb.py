'''

  Check if SVD decomposition is good enough.
  Use iterative PCA to add more elements.

'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import argparse
#sys.path.append('../mbhbs/')
from data_generation.gbs.gb_model import GB_gpu
#from mbhb_model_Lframe import MBHB_gpu

from flow.utils.torchutils import *
from flow.networks.resnet import ResidualNet
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
def iPCA(gcb, dtype, maxA, maxE):

    num_iterations = 1000

    batch_size = 8192
    ipca = IncrementalPCA(n_components=2048, batch_size = batch_size)
 
    for i in range(num_iterations):

       A, E = gb.freqwave_AET(batch_size)
       waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype), torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)
       ipca.partial_fit(waveform) 
       #print(ipca.components_) 
       print('Singular values:')
       print(ipca.singular_values_)
       print('Variance ratios:')
       print(ipca.explained_variance_ratio_)
       print('Cumsum:')
       print(cp.cumsum(ipca.explained_variance_ratio_))
       print(' ')
       if i % 1 == 0:
           A, E = gb.freqwave_AET(16)
           waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype), torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)
           #mbhb_ts = mbhb.timewave_AET()
           waveform_reconstruct = torch.as_tensor(ipca.inverse_transform(ipca.transform(waveform))).type(dtype)
           loss = overlap_tensor(waveform, waveform_reconstruct)  
          
           print('loss.shape = ', loss.shape)             
           print('overlap = ', loss)
           print('i [{}/{}]'.format(i + 1, num_iterations))


# Autoencoder with attention



# Autoencoder with RESNET
class AE_resnet(nn.Module):
    def __init__(self, Nt):
        super(AE_resnet, self).__init__()
        self.encoder = nn.Sequential(
            ResidualNet(
            in_features = Nt,
            out_features = 4096,
            hidden_features = 4096,
            context_features = None,
            num_blocks = 4,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 4096,
            out_features = 2048,
            hidden_features = 2048,
            context_features = None,
            num_blocks = 4,
            activation = F.elu, 
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 2048,
            out_features = 1024,
            hidden_features = 1024,
            context_features = None,
            num_blocks = 4,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 1024,
            out_features = 512,
            hidden_features = 512,
            context_features = None,
            num_blocks = 8,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 512,
            out_features = 256,
            hidden_features = 256,
            context_features = None,
            num_blocks = 16,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True)
        )
        self.decoder = nn.Sequential(
            ResidualNet(
            in_features = 256,
            out_features = 512,
            hidden_features = 256,
            context_features = None,
            num_blocks = 16,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 512,
            out_features = 1024,
            hidden_features = 512,
            context_features = None,
            num_blocks = 8,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 1024,
            out_features = 2048,
            hidden_features = 1024,
            context_features = None,
            num_blocks = 4,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 2048,
            out_features = 4096,
            hidden_features = 2048,
            context_features = None,
            num_blocks = 4,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True),
            ResidualNet(
            in_features = 4096,
            out_features = Nt,
            hidden_features =  4096,
            context_features = None,
            num_blocks = 4,
            activation = F.elu,
            dropout_probability = 0.0,
            use_batch_norm = True) 
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Simple autoencoder to see if we can compress the data well enough  
class AE(nn.Module):
    def __init__(self, Nt):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(Nt, 2048),
            nn.ELU(True),
            nn.Linear(2048, 1024),
            nn.ELU(True),
            nn.Linear(1024, 1024),
            nn.ELU(True), 
            nn.Linear(1024, 512),
            nn.ELU(True),
            nn.Linear(512, 256),
            nn.ELU(True),
            nn.Linear(256, 128),
            nn.ELU(True),
            nn.Linear(128, 128))
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(True),
            nn.Linear(128, 256),
            nn.ELU(True),
            nn.Linear(256, 512),
            nn.ELU(True),
            nn.Linear(512, 1024),
            nn.ELU(True),
            nn.Linear(1024, 1024),
            nn.ELU(True), 
            nn.Linear(1024, 2048),
            nn.ELU(True),
            nn.Linear(2048, Nt))
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



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


def train_AE(gb, dev, dtype, config, maxA, maxE):
    
    model = AE_resnet(config['model']['context']['coeffs']).to(dev)
    criterion = nn.MSELoss()

    learning_rate = 5e-4

    num_epochs = 500000
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100000, eta_min=0, last_epoch=-1)

    for epoch in range(num_epochs):
        # input data with the noise, output data without the noise (will it not corrupt the noise estimation?)
        # otherwise we can add the white noise to the reduced representation of the data

        Nsamples = 2048
        A, E, A_non, E_non = gb.freqwave_AET(Nsamples)
        waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype), torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)
        waveform_non = torch.cat((torch.as_tensor(cp.real(A_non)/maxA).type(dtype), torch.as_tensor(cp.real(E_non)/maxE).type(dtype), torch.as_tensor(cp.imag(A_non)/maxA).type(dtype), torch.as_tensor(cp.imag(E_non)/maxE).type(dtype)), 1)
 
        model.train()
        output = model(waveform)

        plotting = False
        if plotting == True:

            print('waveform.shape = ', waveform.shape)
            plt.figure()
            plt.plot(waveform[0,:].detach().cpu().numpy())
            plt.savefig('wf_'+ str(epoch) +'.png')   
       
            plt.figure()
            plt.plot(cp.real(A[0,:]).get())
            plt.savefig('Ar_nonorm'+ str(epoch) +'.png')

            plt.figure()
            plt.plot(cp.imag(A[0,:]).get())
            plt.savefig('Aim_nonorm'+ str(epoch) +'.png')
 
            plt.figure()
            plt.plot(cp.real(E[0,:]).get())
            plt.savefig('Er_nonorm'+ str(epoch) +'.png')
 
            plt.figure()
            plt.plot(cp.imag(E[0,:]).get())
            plt.savefig('Eim_nonorm'+ str(epoch) +'.png')
 
        loss = criterion(output, waveform_non)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
   
        # Check overlap for reconstructed waveform
        if epoch % 1000 == 0:
            print('epoch [{}/{}], loss:{:.10f}'.format(epoch + 1, num_epochs, loss.data))
            model.eval()

            A, E, A_non, E_non = gb.freqwave_AET(16)
            waveform = torch.cat((torch.as_tensor(cp.real(A)/maxA).type(dtype), torch.as_tensor(cp.real(E)/maxE).type(dtype), torch.as_tensor(cp.imag(A)/maxA).type(dtype),  torch.as_tensor(cp.imag(E)/maxE).type(dtype)), 1)
            waveform_non = torch.cat((torch.as_tensor(cp.real(A_non)/maxA).type(dtype), torch.as_tensor(cp.real(E_non)/maxE).type(dtype), torch.as_tensor(cp.imag(A_non)/maxA).type(dtype),  torch.as_tensor(cp.imag(E_non)/maxE).type(dtype)), 1)
 
            output_test = model(waveform)
            with torch.no_grad():
                loss = criterion(output_test, waveform_non) 
                print('loss = %.10f' % loss)
                #print('overlap = ', overlap(mbhb_ts_scale[0,:], output_test[0,:]))
                print('overlap = ', overlap_tensor(waveform_non, output_test))
                plt.figure()
                plt.plot(output_test[0,:].detach().cpu().numpy()) 
                plt.plot(waveform_non[0,:].detach().cpu().numpy())
                plt.savefig('Compare' + str(epoch) + '.png')
                plt.close()

                for param_group in optimizer.param_groups:
                  print( param_group['lr'])

def main(parser):

    # Parse command line arguments
    parser.add_argument('--config', type=str, default='../../configs/gbs/gb_resample.yaml',
                        help='Path to config file specifying model architecture and training procedure')
    parser.add_argument('--config_data', type=str, default='../../configs/gbs/gb_test.yaml',
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
   
    # Initialise the class for GB waveforms
    gb = GB_gpu(config, config_data, dtype)
    
    A, E, _, _ = gb.freqwave_AET(1000)

    max_Ar_batch = cp.amax(cp.real(A))
    max_Er_batch = cp.amax(cp.real(E))
    max_Aim_batch = cp.amax(cp.imag(A))
    max_Eim_batch  = cp.amax(cp.imag(E))

    dt = config_data['tvec']['dt']
    maxA = cp.sqrt(max_Ar_batch**2 + max_Aim_batch**2)/cp.sqrt(2.0*dt)
    maxE = cp.sqrt(max_Er_batch**2 + max_Eim_batch**2)/cp.sqrt(2.0*dt)

    method = 'AE'

    if method == 'PCA':
        iPCA(gb, dtype, maxA, maxE)
    elif method == 'AE':
        train_AE(gb, dev, dtype, config, maxA, maxE)
    else:
        print('No such method')

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


















