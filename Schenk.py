import numpy as np
import scipy as sp
from scipy.constants import epsilon_0, c, m_e, m_p, e, pi
from scipy.special import i0, i1, iv
from scipy.integrate import quad, dblquad
from joblib import Parallel, delayed
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sbs

from tune_calculation import *
MAX_INTEGRAL_LIMIT = 16
EPSILON = 1e-6
@np.vectorize
def Q_detuning(phi: float, Jz: float, dQmax = 0.001):
    return dQmax*np.exp(-0.5*Jz*(1-np.cos(2*phi)))
@np.vectorize
def Q_average_detuning(Jz: float, dQmax = 0.001):
    return dQmax*np.exp(-0.5*Jz)*i0(0.5*Jz)
@np.vectorize
def B_integrand(phi: float, Jz: float,dQmax = 0.001):
    return Q_detuning(phi, Jz, dQmax) - Q_average_detuning(Jz, dQmax)
@np.vectorize
def B(func: object, Jz: float, phi: float):
    return quad(func, 0, phi, args=(Jz,))[0]
def H_integrand(phi: float, Jz: float, p: int, l: int):
    return np.exp(1j*l*phi)*np.exp(-1j*omega_p/c*np.sqrt(2*Jz*beta_z)*np.cos(phi))*np.exp(-1j/omega_s*B(Jz, phi))
def H(r: float, p: int, l: int):
    return 1./(2*np.pi)*quad()

if __name__ == '__main__':
    Jz = 1.0
    assert (Q_average_detuning(Jz) - 1/(2*np.pi)*quad(Q_detuning, 0, 2*np.pi, args=(Jz,))[0] < EPSILON), 'Detuning implemented incorrectly'
    # Jz = np.random.exponential(scale=1.0, size=(1000, ))
    # phi = np.random.uniform(0., 2*np.pi)
    phi = np.linspace(0, 2*np.pi, 1000)
    for Jz in range(0, 4):
        Bphi = B(B_integrand, Jz, phi)
        plt.plot(phi, Bphi)
    plt.show()