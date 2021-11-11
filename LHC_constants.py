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
BETA_Z = 815.6
Q_X = 60.28
