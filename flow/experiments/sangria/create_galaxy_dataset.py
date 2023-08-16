import h5py
import numpy as np

training_file = 'LDC2_sangria_training_v2.h5'
fid = h5py.File(training_file)
names_v = fid["sky/vgb/cat"].dtype.names
names_d = fid["sky/dgb/cat"].dtype.names
names_i = fid["sky/igb/cat"].dtype.names

print('names_v = ', names_v)
print('names_d = ', names_d)
print('names_i = ', names_i)

#units = [(k, fid['sky/vgb/cat'].attrs.get(k)) for k in names]
#units = dict(units)

params_v = [fid["sky/vgb/cat"][name] for name in names_v]
params_d = [fid["sky/dgb/cat"][name] for name in names_d]
params_i = [fid["sky/igb/cat"][name] for name in names_i]
fid.close()

#print(params_v.shape)
amp = np.expand_dims(np.r_[params_d[1][:,0], params_i[1][:,0], params_v[1][:,0]], axis = 1)
beta =  np.expand_dims(np.r_[params_d[2][:,0], params_i[2][:,0], params_v[4][:,0]], axis = 1)
lam =  np.expand_dims(np.r_[params_d[3][:,0], params_i[3][:,0], params_v[5][:,0]], axis = 1)
f0 =  np.expand_dims(np.r_[params_d[4][:,0], params_i[4][:,0], params_v[9][:,0]], axis = 1)
fdot =  np.expand_dims(np.r_[params_d[5][:,0], params_i[5][:,0], params_v[10][:,0]], axis = 1)
#iota =  np.expand_dims(np.r_[params_d[6][:,0], params_i[6][:,0], params_v[13][:,0]], axis = 1)
#phi0 =  np.expand_dims(np.r_[params_d[7][:,0], params_i[7][:,0], params_v[14][:,0]], axis = 1)
#psi =  np.expand_dims(np.r_[params_d[8][:,0], params_i[8][:,0], params_v[18][:,0]], axis = 1)


#amp = np.expand_dims(params_d[1][:,0], axis = 1)
#beta =  np.expand_dims(params_d[2][:,0], axis = 1)
#lam =  np.expand_dims(params_d[3][:,0], axis = 1)
#f0 =  np.expand_dims(params_d[4][:,0], axis = 1)
#fdot =  np.expand_dims(params_d[5][:,0], axis = 1)
#iota =  np.expand_dims(params_d[6][:,0], axis = 1)
#phi0 =  np.expand_dims(params_d[7][:,0], axis = 1)
#psi =  np.expand_dims(params_d[8][:,0], axis = 1)


#parameters = np.concatenate((amp, beta, lam, f0, fdot), axis=1)
parameters_sky = np.concatenate((amp, beta, lam), axis=1)
parameters_f = np.concatenate((f0, fdot), axis=1)



# Save the dataset to a file
np.save('galaxy_sky_dist.npy', parameters_sky)
np.save('galaxy_f.npy', parameters_f)



#for i in np.arange(len(names_)):
#
#    print('i = ', i)    
#    print('names = ', params[0][i,0])
#    print('amp = ', params[1][i,0])
#    print('f0 = ', params[9][i,0])
#    print('fdot = ', params[10][i,0])
#    print('phi0 = ', params[14][i,0])
#    print('iota = ', params[13][i,0])
#    print('psi = ', params[18][i,0])
#    print('lam = ', params[5][i,0])
#    print('beta = ', params[4][i,0])
   
