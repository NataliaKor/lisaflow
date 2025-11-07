# Reparameterise extrinsic parameters and return back
import numpy as np
import torch
import corner

def modpi(x):
    print('type(x) = ', type(x))
    return torch.remainder(x, torch.pi)


# Reparameterise for better sampling of phi0 and psi
def transform_params(Amp, cosinc, phi0, psi):

    Amp = Amp/(10**(-22))
    # Normalised amplitude
    Ap = Amp*(1.0 + torch.pow(cosinc,2))
    Ac = 2.*Amp*cosinc
    
    a1 =  Ap * (torch.cos(phi0) * torch.cos(2*psi)) +  Ac * (torch.sin(phi0) * torch.sin(2.*psi))
    a2 =  Ap * (torch.sin(2.*psi) * torch.cos(phi0)) - Ac * (torch.cos(2.*psi) * torch.sin(phi0))
    a3 = -Ap * (torch.cos(2.*psi) * torch.sin(phi0)) + Ac * (torch.sin(2.*psi) * torch.cos(phi0))
    a4 = -Ap * (torch.sin(phi0) * torch.sin(2.*psi)) - Ac * (torch.cos(phi0) * torch.cos(2.*psi))

    #a1_ = (a1 + 1.0)/2.0
    #a2_ = (a2 + 1.0)/2.0
    #a3_ = (a3 + 1.0)/2.0
    #a4_ = (a4 + 1.0)/2.0

    return a1.view(-1,1), a2.view(-1,1), a3.view(-1,1), a4.view(-1,1)

# Return back to physical parameters
def reconstruct_params(a1, a2, a3, a4):

    #a1 = -1.0 + 2.0*a1_
    #a2 = -1.0 + 2.0*a2_
    #a3 = -1.0 + 2.0*a3_
    #a4 = -1.0 + 2.0*a4_
 
    A = a1**2 + a2**2 + a3**2 + a4**2
    D = a1*a4 - a2*a3

    Ap = 0.5*(torch.sqrt(A + 2.0*D) + torch.sqrt(A - 2.0*D))
    Ac = 0.5*(torch.sqrt(A - 2.0*D) - torch.sqrt(A + 2.0*D))

    Amp =    0.5*(Ap + torch.sqrt(Ap*Ap - Ac*Ac))
    cosinc = 0.5*Ac/Amp
    phi0 =  -0.5*torch.arctan2(2.0*(a1*a3 + a2*a4), (a1*a1 + a2*a2 - a3*a3 -a4*a4))
    psi =   0.25*torch.arctan2(2.0*(a1*a2 + a3*a4), (a1*a1 + a3*a3 - a2*a2 -a4*a4))

    psi  = torch.where(psi  > 0, psi , psi  + np.pi)
    phi0 = torch.where(phi0 > 0, phi0, phi0 + 2.0*np.pi)

    # psi degeneracy with 
    # psi - 0.5*np.pi
 
    # !!!! Do not forget to renormalise the amplitude back
    return Amp, cosinc, psi, phi0

def transform_params_mbhb(params, direction, injdist, dtype):
    '''
    This code is from Sylvain Marsat's lisabeta.
    Forward and inverse transformation.
    (Simple response 22)
https://github.com/SylvainMarsat/lisabeta/blob/master/lisabeta/inference/inference.py
    '''
    
    if direction == 'forward':
        '''
        Transfrom from the physical parameters to the map parameters.
        '''
        lambda_a = params['lam'] - np.pi/6

        if (np.any(params['dist'] <= 0.) or np.any(params['inc'] < 0.) or np.any(params['inc'] > np.pi) or np.any(params['beta'] < -np.pi/2) or np.any(params['beta'] > np.pi/2)):
            raise ValueError('params are outside physical range.')

        # injdist is just the scale by which we can normalise distance
        d = params['dist'] / injdist
        tiota = np.tan(params['inc']/2)
        ttheta = np.tan(1./2 * (np.pi/2 - params['beta']))
        rho = 1./(4.*d) * np.sqrt(5./np.pi) * 1./(1. + tiota**2)**2 * 1./(1 + ttheta**2)**2

        sigma_plus = rho*np.exp(2.*1j*params['phi_ref']) * (ttheta**4 * np.exp(-2*1j*params['psi']) + tiota**4 * np.exp(2*1j*params['psi'])) * np.exp(-2*1j*lambda_a)
        sigma_minus = rho*np.exp(2*1j*params['phi_ref']) * (np.exp(-2*1j*params['psi']) + ttheta**4 * tiota**4 * np.exp(2*1j*params['psi'])) * np.exp(2*1j*lambda_a)

        if torch.jit.isinstance(params['psi'], dtype): # if torch.jit.isinstance(psi, dtype):
            indexpsi = torch.zeros(params['psi'].shape, dtype = dtype)
            indexpsi[modpi(params['psi']) > np.pi/2] = 1
        else:
            if modpi(params['psi']) <= np.pi/2:
                indexpsi = 0
            else:
                indexpsi = 1

        params_map = {} 
        params_map['Aplus'] = np.abs(sigma_plus)
        params_map['Aminus'] = np.abs(sigma_minus)
        params_map['Phiplus'] = np.angle(sigma_plus)
        params_map['Phiminus'] = np.angle(sigma_minus)
        params_map['lambda'] = lambda_a + np.pi/6
        params_map['sbeta'] = np.sin(params['beta'])
        indexpsi = indexpsi

        #params_map['frame'] = params[frame] # keep track in which frame we are working

        return params_map


    elif direction == 'inverse':
        '''
        Transfrom form the map parameters to the original parameters.
        '''
        Aplus = params_map['Aplus']
        Aminus = params_map['Aminus']
        Phiplus = params_map['Phiplus']
        Phiminus = params_map['Phiminus']
        lambd = params_map['lambda']
        sbeta = params_map['sbeta']
        indexpsi = params_map['indexpsi']

        if (np.any(Aplus <= 0.) or np.any(Aminus <= 0.) or np.any(sbeta<-1.) or torch.any(sbeta>1.) or not torch.all((indexpsi==0) | (indexpsi==1))):
            raise ValueError('params_map are outside physical range.')

        lambda_a = lambd - np.pi/6
        sigma_plus = Aplus * np.exp(1j*Phiplus)
        sigma_minus = Aminus * np.exp(1j*Phiminus)
        rtilde = sigma_plus / sigma_minus * np.exp(4*1j*lambda_a)

        ctheta = sbeta
        a = ((ctheta-1) / (ctheta+1))**2
        bz = (a - rtilde) / (a*rtilde - 1)
        b = np.abs(bz)
        tiota2 = np.sqrt(b)
        ciota = (1 - tiota2) / (1 + tiota2)
        fourpsi = np.angle(bz)
        # Ambiguity in psi: psi0 in [0, pi/2], psi1=psi0+pi/2 in [pi/2, pi]
        # We use indexpsi=[0,1] to represent this degeneracy
        psi0 = pytools.mod_interval(1./4*fourpsi, interval=[0, np.pi/2])
        psi = psi0 + indexpsi*np.pi/2

        # Get a,b,z
        a = ((ctheta-1) / (ctheta+1))**2
        b = ((ciota-1) / (ciota+1))**2
        z = np.exp(1j*4*psi)

        # No ambiguity in phi, since for 22-mode only we can restrict it mod pi
        phi = 1./2 * np.angle(sigma_minus / (np.exp(-2*1j*psi) * (1+ a*b*z) * np.exp(1j*2*lambda_a)))
        phi = pytools.modpi(phi)

        # Amplitude, unambiguous
        rho = np.abs(sigma_minus / (np.exp(-2*1j*psi) * (1+ a*b*z) * np.exp(1j*2*lambda_a)))
        ttheta2 = np.sqrt(a)
        d = 1/(4*rho) * np.sqrt(5/np.pi) * 1./((1+ttheta2)**2 * (1+tiota2)**2)
        beta = np.arcsin(sbeta)
        inc = np.arccos(ciota)

        params = {}
        params['dist'] = d * injdist
        params['inc'] = inc
        params['phi_ref'] = phi
        params['lam'] = lambd
        params['beta'] = beta
        params['psi'] = psi

        #params['Lframe'] = params_map.get('Lframe', False)
        return params

    else:

        print('Wrong direction')




