import lisabeta.lisa.ldctools as ldctools
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
import lisabeta.lisa.lisa as lisa


t_ref = 24960000.0 
bet = 0.29269632679489654            
lam = 3.5091                                 
phi = 1.65330190000514    
the = 1.8554                      
inc = 1.22453213 
  

# Calculate true values of beta and lambda in the LISA reference frame
psi, inc = ldctools.AziPolAngleL2PsiIncl(bet, lam, the, phi)

print('inc = ', inc)

print('.........................')

tL_ref, lamL, betL, psiL = lisa.lisatools.ConvertSSBframeParamsToLframe(t_ref, lam, bet, psi, 0.0)

print('tL_ref = ', tL_ref)
print('lamL = ', lamL)
print('betL = ', betL)
print('psiL = ', psiL)


t_ref_new, lam_new, bet_new, psi_new = lisa.lisatools.ConvertLframeParamsToSSBframe(tL_ref, lamL, betL, psiL, 0.0)

print('t_ref = ', t_ref)
print('t_ref_new = ', t_ref_new)
print('lam = ', lam)
print('lam_new = ', lam_new)
print('bet = ', bet)
print('bet_new = ', bet_new)
print('psi = ', psi)
print('psi_new = ', psi_new)














