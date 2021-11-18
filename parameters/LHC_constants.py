from scipy.constants import m_p, c, e, pi
from numpy import sqrt
SIGMA_Z = 0.06
E_KIN = 6.5e12
GAMMA = 1 + E_KIN * e / (m_p * c**2)
BETA = sqrt(1-GAMMA**-2)
CIRCUMFERENCE = 27e3
OMEGA_REV = 2*pi*BETA*c/(CIRCUMFERENCE)
Q_S = 1.74e-3
OMEGA_S = Q_S*OMEGA_REV
Q_X = 62.28
Q_Y = 60.31

ALPHA_0 = [53.83**-2]
GAMMA_T = 1. / sqrt(ALPHA_0)
ETA = ALPHA_0[0]-GAMMA**-2
BETA_Z = ETA*BETA*c/OMEGA_S
print(OMEGA_REV*Q_S)
