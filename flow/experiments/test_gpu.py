import torch


# Choose CPU or GPU
if torch.cuda.is_available():
    dev = "cuda:0"
    dtype = torch.cuda.FloatTensor
else:
    dev = "cpu"
    dtype = torch.FloatTensor

print('device = ', dev)
 
