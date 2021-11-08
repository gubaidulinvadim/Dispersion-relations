import numpy as np
import scipy as sp
from scipy.constants import epsilon_0, c, m_e, m_p, e, pi
from scipy.special import i0, i1, iv, j0, j1, jv
from scipy.integrate import quad, dblquad
from joblib import Parallel, delayed
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sbs

from tune_calculation import *
MAX_INTEGRAL_LIMIT = 16
EPSILON = 1e-6
CIRCUMFERENCE = 27e3
@np.vectorize
def Q_detuning(phi: float, Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.25*Jz*(1-np.cos(2*phi)))


@np.vectorize
def Q_average_detuning(Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.25*Jz)*i0(0.25*Jz)


@np.vectorize
def B_integrand(phi: float, Jz: float, dQmax=0.001):
    return Q_detuning(phi, Jz, dQmax) - Q_average_detuning(Jz, dQmax)


@np.vectorize
def B(func: object, Jz: float, phi: float):
    return quad(func, 0, phi, args=(Jz,))[0]


@np.vectorize
def H_integrand(phi: float, Jz: float, p: int, l: int):
    beta_z = 815.6
    omega_0 = c/CIRCUMFERENCE
    Q_s = 1.74e-3
    Q_X = 60.28
    omega_s = omega_0*Q_s
    omega_p = p*omega_0+Q_X*omega_0+l*omega_s
    return np.exp(1j*l*phi)*np.exp(-1j*omega_p/c*np.sqrt(2*Jz*beta_z)*np.cos(phi))*np.exp(-1j/Q_s*B(B_integrand, Jz, phi))


@np.vectorize
def H(Jz: float, p: int, l: int):
    @np.vectorize
    def real_func(phi):
        return np.real(H_integrand(phi, Jz, p, l))

    @np.vectorize
    def imag_func(phi):
        return np.imag(H_integrand(phi, Jz, p, l))
    return 1./(2*np.pi)*quad(real_func, 0, 2*pi)[0], 1./(2*np.pi)*quad(imag_func, 0, 2*pi)[0]


if __name__ == '__main__':
    Jz = 1.0
    assert (Q_average_detuning(Jz) - 1/(2*pi)*quad(Q_detuning, 0, 2 *
                                                   pi, args=(Jz,))[0] < EPSILON), 'Detuning implemented incorrectly'
    Jz = np.linspace(0, 3, 1000)
    sigma_z = 0.06
    beta_z = 815.6
    Jz *= sigma_z**2/(2*beta_z)
    Hr, Hi = H(Jz, p=10, l=0)
    plt.plot(Jz, Hr, c='b')
    plt.plot(j0(np.sqrt(2*Jz*beta_z)), c='r')
    plt.plot(Jz, Hi, c='b', linestyle='dashed')

    plt.show()
