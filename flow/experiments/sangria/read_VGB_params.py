import h5py
import numpy as np

training_file = 'LDC2_sangria_training_v2.h5'
fid = h5py.File(training_file)
names = fid["sky/vgb/cat"].dtype.names
units = [(k, fid['sky/vgb/cat'].attrs.get(k)) for k in names]
units = dict(units)
params = [fid["sky/vgb/cat"][name] for name in names]
fid.close()

names = params[0][:,0]
amp = params[1][:,0]
f0 = params[9][:,0]

print('len(names) = ', len(names))

for i in np.arange(len(names)):

    print('i = ', i)    
    print('names = ', params[0][i,0])
    print('amp = ', params[1][i,0])
    print('f0 = ', params[9][i,0])
    print('fdot = ', params[10][i,0])
    print('phi0 = ', params[14][i,0])
    print('iota = ', params[13][i,0])
    print('psi = ', params[18][i,0])
    print('lam = ', params[5][i,0])
    print('beta = ', params[4][i,0])
    print('                ')

