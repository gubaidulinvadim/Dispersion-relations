from scipy.constants import m_p, c, e, pi
from numpy import sqrt, abs
SIGMA_Z = 0.25*58
E_KIN = 2e8
GAMMA = 1 + E_KIN * e / (m_p * c**2)
BETA = sqrt(1-GAMMA**-2)
ETA = -0.67
CIRCUMFERENCE = 1083.6
OMEGA_REV = 2*pi*BETA*c/(CIRCUMFERENCE)
Q_S = 4.47e-3
OMEGA_S = Q_S*OMEGA_REV
BETA_Z = abs(ETA)*BETA*c/(OMEGA_S)
Q_X = 18.86
