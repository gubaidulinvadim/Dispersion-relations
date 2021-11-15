from LHC_constants import *
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
import time
from tune_calculation import *
# MAX_INTEGRAL_LIMIT = 16
# EPSILON = 1e-6
@np.vectorize
def Q_detuning(phi: float, Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.25*Jz*(1-np.cos(2*phi)))


@np.vectorize
def Q_average_detuning(Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.25*Jz)*i0(0.25*Jz)


@np.vectorize
def B_integrand(phi: float, Jz: float, dQmax=.002):
    return Q_detuning(phi, Jz, dQmax) - Q_average_detuning(Jz, dQmax)


@np.vectorize
def B(func: object, Jz: float, phi: float):
    return quad(func, 0, phi, args=(Jz,))[0]


@np.vectorize
def H_integrand(phi: float, Jz: float, p: int, l: int):
    # omega_p = p*OMEGA_REV+Q_X*OMEGA_REV+l*OMEGA_S
    JzB = Jz/(SIGMA_Z**2/(2*BETA_Z))
    # *np.exp(-1j*omega_p/c*np.sqrt(2*Jz*BETA_Z)*np.cos(phi))
    return np.exp(1j*l*phi) * np.exp(-1j/Q_S*B(B_integrand, JzB, phi))


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
    sbs.set(rc={'figure.figsize': (8.3, 5.2),
                'text.usetex': True,
                'font.family': 'serif',
                'font.size': 20,
                'axes.linewidth': 2,
                'lines.linewidth': 3,
                'legend.fontsize': 16,
                'legend.numpoints': 1, },
            style='ticks',
            palette='colorblind',
            context='talk')
    # Jz = 1.0
    # assert (Q_average_detuning(Jz) - 1/(2*pi)*quad(Q_detuning, 0, 2 *
    #    pi, args=(Jz,))[0] < EPSILON), 'Detuning implemented incorrectly'
    # Jz = np.linspace(0, 3, 20)
    # SIGMA_Z = 0.06
    # BETA_Z = 815.6
    # Jz *= SIGMA_Z**2/(2*BETA_Z)
    sbs.set_palette('colorblind')
    # print(max(Jz), 3*SIGMA_Z**2/(2*BETA_Z))
    # phi = np.linspace(0, 2*pi, 1000)
    Jz = np.linspace(0, 90, 1000)
    dQmax = 2e-3
    phi = .25*pi
    plt.plot(Jz, B(B_integrand, Jz, phi)/dQmax)
    print()
    print('max value of the B function:', max(B(B_integrand, Jz, phi)/dQmax))
    time_start = time.process_time()
    p = 0
    l = 0
    print('Order of magnitude for longitudinal amplitude: ',
          SIGMA_Z/(BETA*c)*OMEGA_REV)

    # Hr, Hi = H(Jz, p=p, l=l)
    time_elapsed = time.process_time()-time_start
    print('Time elapsed: {0:.2e}'.format(time_elapsed))
    # plt.plot(Jz/(SIGMA_Z**2/(2*BETA_Z)), np.sqrt(Hr**2+Hi**2), c='b',
    #  linewidth=2, marker='o', markersize=2, label='$|H^0_l(J_z)|$')

    omega_p = p*OMEGA_REV+Q_X*OMEGA_REV+l*OMEGA_S
    # plt.plot(Jz/(SIGMA_Z**2/(2*BETA_Z)),
    #  np.abs(jv(l, np.sqrt(2*Jz*BETA_Z)/c*(omega_p))), c='r', linewidth=1, marker='o', markersize=1, label='$|J_l(J_z)|$')
    # plt.plot(Jz/(SIGMA_Z**2/(2*BETA_Z)), Hi,
    #  c='b', linestyle='dashed', marker='o', markersize=1, alpha=0.5)
    plt.xlabel('$J_z/\epsilon_z$')
    plt.ylabel('Specrtal function')
    plt.legend(frameon=False)
    plt.savefig('Results/'+'2H_{0:}.pdf'.format(l), bbox_inches='tight')
    plt.show()
