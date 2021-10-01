import numpy as np
import scipy as sp
import scipy.stats as ss
from matplotlib import pyplot as plt
import seaborn as sbs
from tune_calculation import *

sbs.set(rc={'figure.figsize':(8.3,5.2)}, style='ticks', palette='colorblind', context='talk')
Ekin = 2e8
n_particles = int(1e5)
p0 = Ekin*e/c
gamma = 1+Ekin*e/(m_p*c**2)
beta = np.sqrt(1-gamma**-2)
K3 = 50
epsnx = 0.25*48e-6#0.25*35e-6
epsny = 0.25*48e-6#0.25*15e-6
a1 = get_octupole_coefficients(5.8, 17.2, K3, 6, 0.75) 
a2 = get_octupole_coefficients(17.7, 5.8, K3, 6, 0.75)
a = epsnx*a1+epsny*a2
Jx = trunc_exp_rv(0, 1.5**2, 1.0, n_particles)
Jy = trunc_exp_rv(0, 1.5**2, 1.0, n_particles)
J = np.array((Jx, Jy))
dQ_oct = get_octupole_tune(a, J)
dQx, dQy = dQ_oct
dQrmsx = np.sqrt(np.var(dQx))
dQrmsy = np.sqrt(np.var(dQy))
print('RMS tune spread values: ({0:.3f}, {1:.3f})'.format(dQrmsx, dQrmsy))
plot_spread(dQx, dQy, normalise=False, filename='SIS100oct')