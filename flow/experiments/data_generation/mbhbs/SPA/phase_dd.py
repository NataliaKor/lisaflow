

# From LALSimInspiralPNCoefficients.c
# The phasing function for TaylorF2 frequency-domain waveform.
# This function is tested in ../test/PNCoefficients.c for consistency
# with the energy and flux in this file.
#
def PNPhasing_F2():
	#PNPhasingSeries *pfa, /**< \todo UNDOCUMENTED */
	#const double m1, /**< Mass of body 1, in Msol */
	#const double m2, /**< Mass of body 2, in Msol */
	#const double chi1L, /**< Component of dimensionless spin 1 along Lhat */
	#const double chi2L, /**< Component of dimensionless spin 2 along Lhat */
	#const double chi1sq,/**< Magnitude of dimensionless spin 1 */
	#const double chi2sq, /**< Magnitude of dimensionless spin 2 */
	#const double chi1dotchi2 /**< Dot product of dimensionles spin 1 and spin 2 */
	#)

    mtot = m1 + m2;
    d = (m1 - m2) / (m1 + m2);
    eta = m1*m2/mtot/mtot;
    m1M = m1/mtot;
    m2M = m2/mtot;
    # Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
    # We write dSigmaL for their (\delta m/m) * \Sigma_\ell
    # There's a division by mtotal^2 in both the energy and flux terms
    # We just absorb the division by mtotal^2 into SL and dSigmaL
    #
    SL = m1M*m1M*chi1L + m2M*m2M*chi2L;
    dSigmaL = d*(m2M*chi2L - m1M*chi1L);

    pfaN = 3.L/(128.L * eta);

    # Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18 */
    pfa->v[0] = 1.L;
    pfa->v[1] = 0.L;
    pfa->v[2] = 5.L*(743.L/84.L + 11.L * eta)/9.L;
    pfa->v[3] = -16.L*PI;
    pfa->v[4] = 5.L*(3058.673L/7.056L + 5429.L/7.L * eta
                     + 617.L * eta*eta)/72.L;
    pfa->v[5] = 5.L/9.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->vlogv[5] = 5.L/3.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->v[6] = (11583.231236531L/4.694215680L
                     - 640.L/3.L * PI * PI - 6848.L/21.L*GAMMA)
                 + eta * (-15737.765635L/3.048192L
                     + 2255./12. * PI * PI)
                 + eta*eta * 76055.L/1728.L
                 - eta*eta*eta * 127825.L/1296.L;
    pfa->v[6] += (-6848.L/21.L)*log(4.);
    pfa->vlogv[6] = -6848.L/21.L;
    pfa->v[7] = PI * ( 77096675.L/254016.L
                     + 378515.L/1512.L * eta - 74045.L/756.L * eta*eta);

    double qm_def1=1.0;
    double qm_def2=1.0;

    /* Compute 2.0PN SS, QM, and self-spin */
    // See Eq. (6.24) in arXiv:0810.5336
    // 9b,c,d in arXiv:astro-ph/0504538
    double pn_sigma = eta * (721.L/48.L*chi1L*chi2L - 247.L/48.L*chi1dotchi2);
    pn_sigma += (720.L*qm_def1 - 1.L)/96.0L * m1M * m1M * chi1L * chi1L;
    pn_sigma += (720.L*qm_def2 - 1.L)/96.0L * m2M * m2M * chi2L * chi2L;
    pn_sigma -= (240.L*qm_def1 - 7.L)/96.0L * m1M * m1M * chi1sq;
    pn_sigma -= (240.L*qm_def2 - 7.L)/96.0L * m2M * m2M * chi2sq;

    double pn_ss3 =  (326.75L/1.12L + 557.5L/1.8L*eta)*eta*chi1L*chi2L;
    pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m1M-120.L*m1M*m1M)*qm_def1 + (-4108.25L/6.72L-108.5L/1.2L*m1M+125.5L/3.6L*m1M*m1M)) *m1M*m1M * chi1sq;
    pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m2M-120.L*m2M*m2M)*qm_def2 + (-4108.25L/6.72L-108.5L/1.2L*m2M+125.5L/3.6L*m2M*m2M)) *m2M*m2M * chi2sq;

    /* Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16 */
    const double pn_gamma = (554345.L/1134.L + 110.L*eta/9.L)*SL + (13915.L/84.L - 10.L*eta/3.L)*dSigmaL;
    int spinorder = SPIN_ORDER_35PN; // Hardwired for simplicity
    switch( spinorder )
    {
        case SPIN_ORDER_ALL:
        case SPIN_ORDER_35PN:
            pfa->v[7] += (-8980424995.L/762048.L + 6586595.L*eta/756.L - 305.L*eta*eta/36.L)*SL - (170978035.L/48384.L - 2876425.L*eta/672.L - 4735.L*eta*eta/144.L) * dSigmaL;
        case SPIN_ORDER_3PN:
            pfa->v[6] += PI * (3760.L*SL + 1490.L*dSigmaL)/3.L + pn_ss3;
        case SPIN_ORDER_25PN:
            pfa->v[5] += -1.L * pn_gamma;
            pfa->vlogv[5] += -3.L * pn_gamma;
        case SPIN_ORDER_2PN:
            pfa->v[4] += -10.L * pn_sigma;
        case SPIN_ORDER_15PN:
            pfa->v[3] += 188.L*SL/3.L + 25.L*dSigmaL;
        case SPIN_ORDER_1PN:
        case SPIN_ORDER_05PN:
        case SPIN_ORDER_0PN:
            break;
    }

    /* At the very end, multiply everything in the series by pfaN */
    for(int ii = 0; ii <= PN_PHASING_SERIES_MAX_ORDER; ii++)
    {
        pfa->v[ii] *= pfaN;
        pfa->vlogv[ii] *= pfaN;
        pfa->vlogvsq[ii] *= pfaN;
    }
}



#
# Step function in boolean version
#
def StepFunc_boolean(t, t1) {
	return (t >= t1)
}


#
# Ansatz for the inspiral phase.
# We call the LAL TF2 coefficients here.
# The exact values of the coefficients used are given
# as comments in the top of this file
# Defined by Equation 27 and 28 arXiv:1508.07253
#
def PhiInsAnsatzInt(Mf, UsefulPowers *powers_of_Mf, PhiInsPrefactors *prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *p 
 
  # m1, /**< Mass of body 1, in Msol */
  # m2, /**< Mass of body 2, in Msol *
  # chi1L, /**< Component of dimensionless spin 1 along Lhat */
  # chi2L, /**< Component of dimensionless spin 2 along Lhat */
  # chi1sq,/**< Magnitude of dimensionless spin 1 */
  # chi2sq, /**< Magnitude of dimensionless spin 2 */
  # chi1dotchi2 /**< Dot product of dimensionles spin 1 and spin 2 */

  mtot = m1 + m2
  d = (m1 - m2) / (m1 + m2)
  eta = m1*m2/mtot/mtot
  m1M = m1/mtot
  m2M = m2/mtot

  SL = m1M*m1M*chi1L + m2M*m2M*chi2L
  dSigmaL = d*(m2M*chi2L - m1M*chi1L)

  # Assemble PN phasing series
  v = xp.power(Mf, 1./3) * xp.power(xp.PI, 1./3)
  logv = xp.log(v)
  #v = powers_of_Mf->third * powers_of_pi.third
  #logv = log(v)
 
  phasing = 5./9. * (7729./84. - 13. * eta) * xp.PI - PI_4

  phasing += xp.PI * ( 77096675./254016. + 378515./1512. * eta - 74045./756. * eta*eta) * powers_of_pi.thirds * powers_of_Mf->two_thirds;
  # SPIN_ORDER_35PN:
  phasing += (-8980424995./762048. + 6586595.*eta/756. - 305.*eta*eta/36.)*SL - (170978035./48384. - 2876425.*eta/672. - 4735.*eta*eta/144.L * dSigmaL


  # prefactors->two_thirds = pn->v[7] * powers_of_pi.two_thirds;

  # prefactors->third = pn->v[6] * powers_of_pi.third;
  # prefactors->third_with_logv = pn->vlogv[6] * powers_of_pi.third;
  # prefactors->logv = pn->vlogv[5];
  # prefactors->minus_third = pn->v[4] / powers_of_pi.third;
  # prefactors->minus_two_thirds = pn->v[3] / powers_of_pi.two_thirds;
  # prefactors->minus_one = pn->v[2] / Pi;
  # prefactors->minus_five_thirds = pn->v[0] / powers_of_pi.five_thirds; // * v^0

  phasing += prefactors->two_thirds * powers_of_Mf->two_thirds;
  phasing += prefactors->third * powers_of_Mf->third;
  phasing += prefactors->third_with_logv * logv * powers_of_Mf->third;
  phasing += prefactors->logv * logv;
  phasing += prefactors->minus_third / powers_of_Mf->third;
  phasing += prefactors->minus_two_thirds / powers_of_Mf->two_thirds;
  phasing += prefactors->minus_one / Mf;
  phasing += prefactors->minus_five_thirds / powers_of_Mf->five_thirds; # * v^0

  # Now add higher order terms that were calibrated for PhenomD
  phasing += ( prefactors->one * Mf + prefactors->four_thirds * powers_of_Mf->four_thirds
			   + prefactors->five_thirds * powers_of_Mf->five_thirds
			   + prefactors->two * powers_of_Mf->two
			 ) / p->eta;

  return phasing





################################################################################
# This function computes the IMR phase given phenom coefficients.
# Defined in VIII. Full IMR Waveforms arXiv:1508.07253
#################################################################################
def IMRPhenDPhase

    # Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    # The inspiral, intermendiate and merger-ringdown phase parts

    # split the calculation to just 1 of 3 possible mutually exclusive ranges

    if (!StepFunc_boolean(f, p->fInsJoin)):	# Inspiral range
    
        PhiIns = PhiInsAnsatzInt(f, powers_of_f, prefactors, p, pn)
	return PhiIns;
  

    if (StepFunc_boolean(f, p->fMRDJoin))	# MRD range
 
	  double PhiMRD = 1.0/p->eta * PhiMRDAnsatzInt(f, p) + p->C1MRD + p->C2MRD * f
	  return PhiMRD
 
    # Intermediate range
    PhiInt = 1.0/p->eta * PhiIntAnsatz(f, p) + p->C1Int + p->C2Int * f
  return PhiInt




















def derivative_phi(M_sec, freq):

    df=1.0e-6/M_sec

    if(freq[0] - df < 0.0):
        df = 0.5*freq[0]
    v = 4.0*xp.PI*df
    u = df*df
    
    for i in range(n):
    
        fp = freq[i] + df
        fm = freq[i] - df
            
        Mfp = M_sec * fp # geometric frequency
        Mfm = M_sec * fm # geometric frequency
            
        # That stuff has to be rewritten
        UsefulPowers powers_of_fp;
        UsefulPowers powers_of_fm;
        status_in_for = init_useful_powers(&powers_of_fp, Mfp);
        status_in_for = init_useful_powers(&powers_of_fm, Mfm);            

        phip = IMRPhenDPhase(Mfp, pPhi, pn, &powers_of_fp, &phi_prefactors)
        phip -= t0 * (Mfp - MfRef) + phi_precalc

        phim = IMRPhenDPhase(Mfm, pPhi, pn, &powers_of_fm, &phi_prefactors);
        phim -= t0 * (Mfm - MfRef) + phi_precalc
            
        time[i] = (phip-phim)/v;
        pdd[i] = (phip+phim-2.0*phase[i])/u;








def get_time():






def get_pdd():










