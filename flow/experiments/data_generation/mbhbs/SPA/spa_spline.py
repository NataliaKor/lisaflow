# Code for SPA from Niel rewritten in python

from abc import ABC, abstractmethod

import math
import numpy as np
import cupy as cp
import cupyx as cpx

from qnm import QNMData_length, QNMData_a, QNMData_fring
from constants import *

# Make a choice if to use np or cp

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase


#
# Equation 20 arXiv:1508.07253 (called f_peak in paper)
# analytic location of maximum of AmpMRDAnsatz
#
def fmaxCalc():

  fRD = fring(eta, chi1, chi2, finspin)
  fDM = fdamp(eta, chi1, chi2, finspin)

  chi = chiPN(eta, chi1, chi2)

  # Compute gamma_i's, rho_i's first then delta_i'
  gamma2 = gamma2_fun(eta, chi)
  gamma3 = gamma3_fun(eta, chi)

  # NOTE: There's a problem with this expression from the paper becoming imaginary if gamma2>=1
  # Fix: if gamma2 >= 1 then set the square root term to zero.
  if (gamma2 <= 1):
    return xp.abs(fRD + (fDM*(-1 + xp.sqrt(1 - pow_2_of(gamma2)))*gamma3)/gamma2)
  else
    return xp.abs(fRD + (fDM*(-1)*gamma3)/gamma2)




# For the choice 
def phase_derivative(phenom):

    # We have to define a cut off frequency

    if (fRef_in == 0.0):
        fRef = xp.min(fmaxCalc()/M_sec , f_max)
    else:
        fRef = 0.0

    MfRef = M_sec * fRef
    
    phenom(m1,
           m2,
           self.chi1,
           self.chi2,
           self.dist,
           phic,
           fRef_in,
           self.tc,
           freqs=fRef)
    phifRef = phenom.phase 

    phi_precalc = 2.*self.phi0 + phifRef 

    df = 1.0e-6/M_sec
    if (f[0]-df < 0.0):
        df = 0.5*f[0]
    v = (4.0*xp.PI*df)
    u = df*df
 
    time = xp.zeros(n)
    pdd = xp.zeros(n)
    for i in range(n):
        
        fp = f[i] + df
        fm = f[i] - df

        phenom(m1,
               m2,
               self.chi1,
               self.chi2,
               self.dist,
               phic,
               fRef_in,
               self.tc,
               freqs=fp)

        phip = phenom.phase

        phenom(m1,
               m2,
               self.chi1,
               self.chi2,
               self.dist,
               phic,
               fRef_in,
               self.tc,
               freqs=fm)

        phim = phenom.phase
            
        Mfp = M_sec * fp # geometric frequency
        Mfm = M_sec * fm # geometric frequency
            
        #phip = IMRPhenDPhase(Mfp, pPhi, pn, &powers_of_fp, &phi_prefactors);
        phip -= t0 * (Mfp - MfRef) + phi_precalc
        #phim = IMRPhenDPhase(Mfm, pPhi, pn, &powers_of_fm, &phi_prefactors);
        phim -= t0 * (Mfm - MfRef) + phi_precalc
            
        time[i] = (phip - phim)/v
        pdd[i] = (phip + phim - 2.0*phase[i])/u

    return time, pdd




#########################################################################
# Final Spin and Radiated Energy formulas described in 1508.07250
####################################################################### 
# Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
# s defined around Equation 3.6.
#
def FinalSpin0815_s(eta, s):

    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    s2 = s*s
    s3 = s2*s
    s4 = s3*s

    return 3.4641016151377544*eta - 4.399247300629289*eta2 +i \
           9.397292189321194*eta3 - 13.180949901606242*eta4 + \
           (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s + \
           (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 + \
           (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 + \
           (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4


######################################################################
# Wrapper function for FinalSpin0815_s
def FinalSpin0815(eta, chi1, chi2):
  
    # Convention m1 >= m2
    Seta = xp.sqrt(1.0 - 4.0*eta);
    m1 = 0.5 * (1.0 + Seta);
    m2 = 0.5 * (1.0 - Seta);
    m1s = m1*m1;
    m2s = m2*m2;
    # s defined around Equation 3.6 arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2)
    return FinalSpin0815_s(eta, s)


######################################################################
# Formula to predict the total radiated energy. Equation 3.7 and 3.8 arXiv:1508.07250
# Input parameter s defined around Equation 3.7 and 3.8.
#
def EradRational0815_s(eta, s):
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta

  return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 - 1.4756929437702908*eta + 7.304676214885011*eta2)*s)


####################################################################
# Wrapper function for EradRational0815_s.
#
def EradRational0815(eta, chi1, chi2):
    # Convention m1 >= m2
    Seta = xp.sqrt(1.0 - 4.0*eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)

  return EradRational0815_s(eta, s)


##################################################################
# fring is the real part of the ringdown frequency
# 1508.07250 figure 9
#
def fring(eta, chi1, chi2, finspin):

    if (finspin > 1.0):
         except ValueError:
             print("PhenomD fring function: final spin > 1.0 not supported")

    res = cupyx.scipy.interpolate.pchip_interpolate(QNMData_a, QNMData_fring, finspin)
  
    return res / (1.0 - EradRational0815(eta, chi1, chi2))


####################################################################
# fdamp is the complex part of the ringdown frequency
# 1508.07250 figure 9
#
def fdamp(eta, chi1, chi2, finspin):


    if (finspin > 1.0):
         except ValueError:
             print("PhenomD fdamp function: final spin > 1.0 not supported")

    res = cupyx.scipy.interpolate.pchip_interpolate(QNMData_a, QNMData_fdamp, finspin)

    return res / (1.0 - EradRational0815(eta, chi1, chi2))


#
# PN reduced spin parameter
# See Eq 5.9 in http://arxiv.org/pdf/1107.1267v2.pdf
#
def chiPN(eta, chi1, chi2):
    # Convention m1 >= m2 and chi1 is the spin on m1
    delta = xp.sqrt(1.0 - 4.0*eta)
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    return chi_s * (1.0 - eta*76.0/113.0) + delta*chi_a


####################################################################################
# gamma 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
#
def gamma2_fun(eta, chi):
    xi = -1 + chi
    xi2 = xi*xi
    xi3 = xi2*xi
    eta2 = eta*eta

    return 1.010344404799477 + 0.0008993122007234548*eta
        + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2)*xi
        + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi2
        + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi3


#####################################################################################
# gamma 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
#
def gamma3_fun(eta, chi):
    xi = -1 + chi
    xi2 = xi*xi
    xi3 = xi2*xi
    eta2 = eta*eta

    return 1.3081615607036106 - 0.005537729694807678*eta
        + (-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2)*xi
        + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi2
        + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi3


class SPA(ABC):
    def __init__(self, config, config_data, dtype):

         super().__init__()

         self.config_data = config_data
         self.config = config
         self.dtype = dtype

         self.dt = config_data['tvec']['dt']
         self.Tobs = config_data['tvec']['Tobs'] 

         self.fRef_in = 0 # = PDfref 

         self.Mc = 0.
         self.Mtot = 0.
         self.chi1 = 0.
         self.chi2 = 0.
         self.phi0 = 0.
         self.tc =  0.
         self.dist =  0.

         self.freqs = None
         self.param = None

    def SetUp(self, NFmax):
        '''
          This function sets up the frequency sample array, the time-frequency map and finds the PhenomD amplitude and phase.
          NFmax -- the size of the holder arrays
          NFS -- the actual size
          log in Mc, Mt, DL ?
        '''
        # These parameters we have also converted 
        # distance =   np.exp(params[6])*1.0e9*PC_SI # distance
        #Mc =        xp.exp(params[0])*TSUN
        #Mtot =      xp.exp(params[1])*TSUN

        FF = xp.zeros(NFmax)

        self.eta = xp.power((self.Mc_s/self.Mtot_s), (5.0/3.0)) # symmetric mass eario

        # eta is a symmetric mass ratio
        if self.eta > 0.25:
            dm = 0.0
        else
            dm = xp.sqrt(1.0-4.0*self.eta)

        m1 = Mtot_s*(1.0+dm)/2.0
        m2 = Mtot_s*(1.0-dm)/2.0
        
        # Check what is passed as input here
        tstart = 0.0
        fmin, fmax, frg = self.StartStop(tstart, self.Tobs)
    
        # Because of the way the LDC phase is set at merger, we have to take the signal all the way out to
        # the merger frequency even if the obsevation time doesn't get us to merger. The signal is still
        # truncated by the window at Tend, so the SNRs etc will be correct
    
        #fmax = 2.0*fr;
    
        # printf("%e %e %e\n", fmin, fmax, 1.0/Mtot);
    
        # this can happen when tc is really small and the masses are large
        if (fmax < fmin):
            fmin = 0.5*fmax
   
        dfmin = 1.0/self.Tobs
        dfmax = fmax/100.0
        DT = 1.0e5
    
        fac = DT*xp.power(8.0*xp.PI, 8.0/3.0)*3.0/40.0*xp.power(self.Mc_s,5.0/3.0)
    
        f = fmin
        NF = 1

        while f < fmax:
   
            df = fac*xp.power(f, 11.0/3.0)
            #printf("%e %e %e %e\n", f, df, dfmin, dfmax);
            if (df < dfmin):
                df = dfmin
            if (df > dfmax):
                df = dfmax
            f += df
            NF += 1
 
        # printf("%d %e\n", NF, f)
        # printf("%d %e %e\n", NF, fmin, fmax)
    
        # Need to catch is NF > NFmax
    
        f = fmin
        FF[0] = fmin
        for i in range(1, NF):
    
            df = fac*xp.power(f, 11.0/3.0)
            if (df < dfmin): 
                df = dfmin
            if (df > dfmax): 
                df = dfmax
            f += df
            FF[i] = f
    
    
        if (NF < 4):
    
            NF = 4
            df = (fmax - fmin)/3.0
            FF[0] = fmin
            for i in range(1, NF):
                FF[i] = fmin + df*i
    
         
    #for (i=0; i< NF; i++) printf("%d %e\n", i, FF[i]);
    
        self.Intrinsic(NF, FF, TF, PF, AF, AT)

        return NF
    


    # FINISHED
    def StartStop(self, tstart, tstop):
        '''
          Find the frequencies which correspond to the starting and stopping time.
        '''     

        # Nyquist frequency
        fny = 1.0/(2.0*self.dt)
   
        af = FinalSpin0815(self.eta, self.chi1, self.chi2)

        # fring is the real part of the ringdown frequency
        fr = fring(self.eta, self.chi1, self.chi2, af)/self.Mtot_s
    
        if fr < 0.0: 
            fr = 1.0/self.Mtot_s  # should not happen
    
        #frg = fr
     
        # Here we space the frequency array to give approximately equal spacing in time
        # The dynamic frequency spacing is capped to be between 1e-6 and 1e-4 Hz

        # guess at fmin (where the signal starts at t=tstart)
        fmin = 0.9*xp.power((xp.power(self.Mc_s,5.0/3.0)*(self.tc-tstart)/5.0),-3.0/8.0)/(8.0*xp.PI)
      
        if fmin < 1.0/self.Tobs:
            fmin = 1.0/self.Tobs
    
        # find the frequency at t = tstart
        i = 0
        tf = self.Tobs  # TODO check if this condition works correctly  # math.inf 
        while (i < 10) and xp.abs(tf - tstart) > 1.0 and fmin == fmin: # I do not understand what the last condition mean

            phic = 0.0
            fnew, tf, Amp, Phase = self.getfreq(tstart, fmin, phic)
            if fnew < 1.0/self.Tobs:
                fnew = 1.0/self.Tobs
            fmin = fnew
            print('fnew = ' + str(fnew) + 'tf = ' + str(tf) + 'tstart = ' + str(tstart))
            i += 1
    
        # nan catcher
        if fmin != fmin: # I do not understand this condition
            fmin = 1.0/self.Tobs
        if fmin < 0.0:
            fmin = 1.0/self.Tobs
    
        print('fr = ', fr)
    
        fmax = 2.0*fr
        i = 0
        if self.tc > tstop:
  
            fmax = 0.9*xp.power((xp.power(Mc,5.0/3.0)*(tc-tstop)/5.0) ,-3.0/8.0)/(8.0*xp.PI)
            if (fmax < fmin):
                fmax = fmin + 1.0/self.Tobs
            # find the frequency at t = tstop
            while (i<10) and xp.abs(tf - tstop) > 1.0:
                phic = 0.0
                fnew, tf, Amp, Phase = self.getfreq(tstop, fmin, phic)
                if (fnew < fmin):
                    fnew = fmin + 1.0/self.Tobs
                fmax = fnew
                # printf("%e %e %e\n", fnew, tf, tstop);
                i += 1
    
        # nan catcher
        if (fmax != fmax):
            fmax = 2.0*fr
        if (fmax > fny):
            fmax = fny
        if (fmax < fmin):
            fmax = 2.0*fmin
    
        # printf("%e %e %e", fmax, fr, fny); 
    
        fstart = fmin
        fstop = fmax
        
        return fstart, fstop, fr
    

    # FINISHED
    def getfreq(self, ts, fguess, phic): 
 
        ep = 1.0e-6/self.Mtot_s
        v = (4.0*xp.PI*ep)
        u = (2.0*xp.PI*ep*ep)
    
        if (fguess - ep < 0.0):
            ep = 0.5*fguess
    
        freq = xp.zeros(3)
        freq[0] = fguess - ep
        freq[1] = fguess
        freq[2] = fguess + ep
    
        #ret = IMRPhenomDGenerateh22FDAmpPhase(&ap,freq,phic,fRef_in,m1_SI, m2_SI, chi1, chi2, distance);
        phenom = PhenomHMAmpPhase(use_gpu=True, run_phenomd=True)
        phenom(self.m1,
               self.m2,
               self.chi1, 
               self.chi2,
               self.dist,
               phic, # phi_ref They hard code it to be 0.0, should it be self.phi0?
               self.fRef_in,
               self.tc, # t_ref
               freqs=freq)

        tnew = (phenom.phase[2] - phenom.phase[0])/v + self.tc
        tf = tnew 

        dtdf = (phenom.phase[2] + phenom.phase[0] - 2.0*phenom.phase[1])/u
        delT = ts - tnew
        delF = delT/dtdf
        fnew = fguess + delF
           
        Amp = h22fac*phenom.amp[1]
        #fonfs = fguess/fstar;
        #x *= 8.0*fonfs*sin(fonfs);   // conversion to fractional frequency and leading order TDI transfer function

        #Amp = x 
        Phase = phenom.phase[1]
    
        return fnew, tf, Amp, Phase 

  
    def Intrinsic(self, NF, FF):
        # NF
        # FF
        # TF
        # PF
        # AF

        PF = xp.zeros(NF)
        AF = xp.zeros(NF)
        AT = xp.zeros(NF)   

        # Make GPU use an option
        amp_phase_FDwf = PhenomHMAmpPhase(use_gpu=True, run_phenomd=True)

        #RealVector *freq
        
        fRef_in = self.PDfref
    
        #distance = xp.exp(params[6])*1.0e9*PC_SI  #  distance
        eta = xp.power((self.Mc_s/self.Mtot_s), (5.0/3.0))
        if (eta > 0.25): 
            dm = 0.0
        else:
            dm = xp.sqrt(1.0 - 4.0*eta)
     
        # TODO check that these are correct units 
        m1 = self.Mtot_s*(1.0+dm)/2.0
        m2 = self.Mtot_s*(1.0-dm)/2.0
        #m1_SI = m1*MSUN_SI/MTSUN_SI
        #m2_SI = m2*MSUN_SI/MTSUN_SI
    
        #chi1 = params[2]  # Spin1
        #chi2 = params[3]  # Spin2
        #tc = params[5]    # merger time
    
        freq = xp.zeros(NF)
    
        for i in range(NF):
            freq[i] = FF[i]
   
        # Can we also pass here the imput frequencies on which we evaluate the waveform 
        amp_phase_FDwf(m1,
                       m2,
                       self.chi1,
                       self.chi2,
                       self.dist,
                       phic,
                       fRef_in,
                       self.tc,
                       freqs=freq)
 
   

        #ret = IMRPhenomDGenerateh22FDAmpPhase(
        #                                  &ap,      /**< [out] FD waveform */
        #                                  freq,    /**< Input: frequencies (Hz) on which to evaluate h22 FD */
        #                                  0.0,                  /**< Orbital phase at fRef (rad) */
        #                                  fRef_in,               /**< reference frequency (Hz) */
        #                                  m1_SI,                 /**< Mass of companion 1 (kg) */
        #                                  m2_SI,                 /**< Mass of companion 2 (kg) */
        #                                  chi1,                  /**< Aligned-spin parameter of companion 1 */
        #                                  chi2,                  /**< Aligned-spin parameter of companion 2 */
        #                                  distance               /**< Distance of source (m) */
        #                                  );
    
        flag = 0
        told = amp_phase_FDwf.tf[0] + self.tc

        for i in range(NF):
    
            PF[i] = amp_phase_FDwf.phase[i]
        
            AF[i] =  h22fac * amp_phase_FDwf.amp[i]

            # fonfs = freq->data[i]/fstar;
            # AF[i] *= (8.0*fonfs*sin(fonfs));   // conversion to fractional frequency and leading order TDI transfer function
        
            # SPA to get the time domain amplitude
            AT[i] = AF[i]*xp.sqrt(2.0*xp.PI/xp.abs(ap->pdd[i]));
        
            t = ap->time[i]+tc;
            if(t < told && flag == 0)
       
                flag = 1;
                ii = i-1;
                tx = told;
       
            TF[i] = t;
            if(t < -Tpad) TF[i] = -Tpad;
            if(flag == 1) TF[i] = tx +(double)(i-ii)*Mtot;
            told = t;
            #printf("%d %e %e\n", i, FF[i], TF[i]);
   


def bwlf(in, out, fwrv, M, n, s, f):

     '''
     Butterworth bandpass filter
     n = filter order 2,4,6,8,...
     s = sampling frequency
     f = half power frequency
     '''
    
    if (n % 2):
         print("Order must be 2,4,6,8,...")
         return
         # I have to make an exceprion here
    
    n = n/2
    a = xp.tan(xp.PI*f/s)
    a2 = a*a

    # Initialise variables with empty arrays
    A = xp.zeros(n)
    d1 = xp.zeros(n)
    d2 = xp.zeros(n)
    w0 = xp.zeros(n)
    w1 = xp.zeros(n)
    w2 = xp.zeros(n)
    
    for i in range(n):
   
        r = xp.sin(xp.PI*(2.0*i+1.0)/(4.0*n))
        s = a2 + 2.0*a*r + 1.0
        A[i] = a2/s
        d1[i] = 2.0*(1-a2)/s
        d2[i] = -(a2 - 2.0*a*r + 1.0)/s
        w0[i] = 0.0
        w1[i] = 0.0
        w2[i] = 0.0
   
    
    for j in range(M):
  
        if fwrv == 1:
            x = in[j]
        if fwrv == -1:
            x = in[M-j-1]
        for i in range(n):
      
            w0[i] = d1[i]*w1[i] + d2[i]*w2[i] + x
            x = A[i]*(w0[i] + 2.0*w1[i] + w2[i])
            w2[i] = w1[i]
            w1[i] = w0[i]
        
        if fwrv == 1:
            out[j] = x
        if fwrv == -1:
            out[M-j-1] = x
    
    
    return


def timearray(tc, freq, N, TF, AmpPhaseFDWaveform *ap):
    '''
      Time array
    '''
       
    for i in range(N-1):
        TF[i] = ( ap->phase[i+1]- ap->phase[i-1])/(2.0*PI*(freq->data[i+1]-freq->data[i-1])) + tc;
    
    TF[0] = TF[1]
    TF[N-1] = TF[N-2]
    
    j = N-1
    flag = 0
    for i in range(N-1): 
        # catch where time turns over
        if(TF[i+1] < TF[i] && flag == 0)
        {
            j = i;
            flag = 1;
        }
        # don't allow time to go too far into the past
        if(TF[i] < -Tpad) TF[i] = -Tpad;
    
    # freeze time at turn over
    for i in arange(N):
        TF[i] = TF[j]
    



  








def main():

    # Goal is to simulate the last portion of a merger. The conditon will ultimtaley be set by the fdot that the fast wavelet code can tolerate. Here looking to start at t ~ 100 M before merger.
  
    #Mtot = (m1_SI + m2_SI)/MSUN_SI;  # total mass in solar masses
    #Mc = pow(m1_SI*m2_SI,3.0/5.0)/pow(m1_SI + m2_SI, 1.0/5.0)/MSUN_SI; # chirp mass in solar masses
 
    # We have as input masses in the Solar mass
    

    # Order of the parameters in the array
    #  [0] ln(Mc)  # For us not in log, but keep it in mind while changing the code
    #  [1] ln(Mt)  # For us not in log, but keep it in mind while changing the code 
    #  [2] Spin1 
    #  [3] Spin2 
    #  [4] phic 
    #  [5] tc 
    #  [6] ln(distance) # For us not in log, but keep iot in mind while changing the code
    #  [7] EclipticCoLatitude # We have Latitude: Colatitude = pi/2 - Latitude
    #  [8] EclipticLongitude  
    #  [9] polarization
    #  [10] inclination
   
    # Order of parameters:
    spa_mbhb_param = xp.vstack((mu, mtot, a1, a2, phi_ref, tc, dist, beta, lam, psi, inc,  psi)).T
    
    self.Mc = spa_mbhb_param[:,0]
    self.Mtot =  spa_mbhb_param[:,1]
    self.chi1 =  spa_mbhb_param[:,2]
    self.chi2 =  spa_mbhb_param[:,3]
    self.phi0 =  spa_mbhb_param[:,4]
    self.tc =  spa_mbhb_param[:,5]
    self.dist =  spa_mbhb_param[:,6] # Already converted to meters

    #params[0] = log(Mc)
    #params[1] = log(Mtot)
    #params[2] = chi1
    #params[3] = chi2
    #params[4] = phi0
    #params[5] = tc
    #params[6] = log(distance)
    
    #Mtot = m1 + m2
    #Mc = xp.power(m1*m2,3.0/5.0)/xp.power(Mtot, 1.0/5.0)

    self.Mtot_s = MTSUN_SI*self.Mtot  # total mass in seconds
    self.Mc_s = MTSUN_SI*self.Mc      # chirp mass in seconds
    
    N = (int)(self.Tobs/self.dt)
    
    #print(f'N = ' + N + ', Tobs = ' + Tobs + ', tc =  ' + tc +  ' Mtot*1000.0 = ',  Mtot*1000.0)
   
    # 
    NFmax = 5000
 
    #SetUp(Tobs, params, NFmax, NF, FF, TF, PF, AF, AT)
    NF, TF, FF, PF, AF, AT = self.SetUp(NFmax)

    xp.savetxt('PhenomD.dat', TF, FF, PF, AF, AT)
    #out = fopen("PhenomD.dat","w");
    #   for (i = 0; i < NF; ++i)
    #   {
    #       fprintf(out,"%.15e %.15e %.15e %.15e %.15e\n", TF[i], FF[i], PF[i], AF[i], AT[i]);
    #   }
    #   fclose(out);
    
    gsl_interp_accel *PTacc = gsl_interp_accel_alloc();
    gsl_spline *PTspline = gsl_spline_alloc (gsl_interp_cspline, NF);
    gsl_spline_init(PTspline, TF, PF, NF);
    
    gsl_interp_accel *PFacc = gsl_interp_accel_alloc();
    gsl_spline *PFspline = gsl_spline_alloc (gsl_interp_cspline, NF);
    gsl_spline_init(PFspline, FF, PF, NF);
    
    gsl_interp_accel *ATacc = gsl_interp_accel_alloc();
    gsl_spline *ATspline = gsl_spline_alloc (gsl_interp_cspline, NF);
    gsl_spline_init(ATspline, TF, AT, NF);
    
    gsl_interp_accel *AFacc = gsl_interp_accel_alloc();
    gsl_spline *AFspline = gsl_spline_alloc (gsl_interp_cspline, NF);
    gsl_spline_init(AFspline, FF, AF, NF);
    
    gsl_interp_accel *FTacc = gsl_interp_accel_alloc();
    gsl_spline *FTspline = gsl_spline_alloc (gsl_interp_cspline, NF);
    gsl_spline_init(FTspline, TF, FF, NF);
    
    out = fopen("PhenomD_tinterp.dat","w");
    for (i = 0; i < N; ++i)
    {
        t = (double)(i)*dt;
        
        if(t > TF[0] && t < TF[NF-1])
        {
        f = gsl_spline_eval (FTspline, t, FTacc);
        A = 0.5*(1.0+tanh((f-(fstart+2.0*fr))/fr));
            
        AA = gsl_spline_eval (ATspline, t, ATacc);
        PP = gsl_spline_eval (PTspline, t, PTacc);
            
        FD = gsl_spline_eval_deriv(FTspline, t, FTacc);
        
        fprintf(out,"%.15e %.15e %.15e %.15e %.15e\n", t, f, PP, AA, FD);
        }
        
    }
    fclose(out);
    
    
    
    fstart = FF[0];
    fr = fstart/8.0;
    
    printf("%e %e\n", fstart, 1.0/Tobs);
    
    double *H;
    H = (double*)malloc(sizeof(double)* (N));
    
    H[0] = 0.0;
    H[N/2] = 0.0;
    out = fopen("PhenomD_freq_h.dat","w");
    for (i=1; i< N/2; i++)
    {
        f = (double)(i)/Tobs;
        
        H[i] = 0.0;
        H[N-i] = 0.0;
        
        if(f > FF[0] && f < FF[NF-1])
        {
           
        A = 0.5*(1.0+tanh((f-(fstart+2.0*fr))/fr));
        
        AA = gsl_spline_eval (AFspline, f, AFacc);
        PP = gsl_spline_eval (PFspline, f, PFacc);
        
        H[i] = A*AA*cos(2.0*PI*f*(Tobs-tc)-PP);
        H[N-i] = A*AA*sin(2.0*PI*f*(Tobs-tc)-PP);
        }
        
        fprintf(out,"%.15e %.15e %.15e\n", f, H[i], H[N-i]);
    }
    fclose(out);
    
    gsl_fft_halfcomplex_radix2_inverse(H, 1, N);
    
    out = fopen("PhenomD_time_h.dat","w");
    for (i = 0; i < N; ++i)
    {
        fprintf(out,"%.15e %.15e\n", (double)(i)*dt, H[i]/(2.0*dt));
    }
    fclose(out);
    
    out = fopen("PhenomD_time_SPA.dat","w");
    for (i = 0; i < N; ++i)
    {
        t = (double)(i)*dt;
        AA = 0.0;
        PP = 0.0;
        A = 0.0;
        
        if(t > TF[0] && t < TF[NF-1])
        {
        f = gsl_spline_eval (FTspline, t, FTacc);
        //A = 0.5*(1.0+tanh((f-(fstart+2.0*fr))/fr));
        A = 1.0;
            
        AA = gsl_spline_eval (ATspline, t, ATacc);
        PP = gsl_spline_eval (PTspline, t, PTacc);
        
        }
        
        fprintf(out,"%.15e %.15e\n", t, A*AA*cos(PP-2.0*PI*f*(t-tc)+PI/4.0));
       
    }
    fclose(out);

    




if __name__=='__main__':
    main()
