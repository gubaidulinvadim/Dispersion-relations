from functools import lru_cache
from tune_calculation import *
import time
import seaborn as sbs
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.special import i0, i1, iv, ive, j0, j1, jv
from scipy.constants import epsilon_0, c, m_e, m_p, e, pi
import scipy as sp
import numpy as np
from parameters.LHC_constants import *
print('Slip factor: ', ETA)
# MAX_INTEGRAL_LIMIT = 16
# EPSILON = 1e-6


def B_sum(phi, J_z, n_max=100):
    n = np.linspace(1, n_max, n_max, dtype=np.int64)
    return np.sum(np.sin(2*np.tensordot(phi, n, axes=0))*ive(n, .5*J_z)/n, axis=1)


def RFQ_sum(phi, J_z, n_max=100):
    n = np.linspace(1, n_max, n_max, dtype=np.int64)
    return np.sum(np.sin(2*np.tensordot(phi, n, axes=0))*jv(2*n, np.sqrt(2*J_z))/n*(-1)**n, axis=1)


phi = np.linspace(0, 2*pi, 2000)
@np.vectorize
# @lru_cache
def H_sum(z_1, z_2, l=0, n_max=20):
    n = np.linspace(-n_max, n_max, 2*n_max+1, dtype=np.int64)
    a = jv(n, z_2)
    b = jv(l-2*n, z_1)
    return 1j**(-l)*np.dot(a, b)


@np.vectorize
def Q_detuning(phi: float, Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.5*Jz*(1-np.cos(2*phi)))


@np.vectorize
def Q_average_detuning(Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.5*Jz)*i0(0.5*Jz)


@np.vectorize
def B_integrand(phi: float, Jz: float, dQmax=.001):
    return Q_detuning(phi, Jz, dQmax) - Q_average_detuning(Jz, dQmax)


@np.vectorize
def Q_detuning_RFQ(phi: float, Jz: float, dQmax=0.001):
    return dQmax*np.cos(np.sqrt(2*Jz)*np.cos(phi))


@np.vectorize
def Q_average_detuning_RFQ(Jz: float, dQmax=0.001):
    return dQmax*j0(np.sqrt(2*Jz))


@np.vectorize
def B_integrand_RFQ(phi: float, Jz: float, dQmax=.001):
    return Q_detuning_RFQ(phi, Jz, dQmax) - Q_average_detuning_RFQ(Jz, dQmax)


@np.vectorize
def Q_detuning_Qpp(phi: float, Jz: float, dQmax=0.001):
    return dQmax/3*Jz - np.cos(2*phi)-dQmax/3*Jz


@np.vectorize
def Q_average_detuning_Qpp(Jz: float, dQmax=0.001):
    return dQmax/3*Jz


@np.vectorize
def B_integrand_Qpp(phi: float, Jz: float, dQmax=.001):
    return -dQmax/3*Jz*np.cos(2*phi)


@np.vectorize
def B_integrand_Qp(phi: float, Jz: float, dQmax=.001):
    return dQmax/np.sqrt(3)*np.sqrt(2*Jz)*np.sin(phi)


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
    # Jz = np.linspace(0, 3, 1000)
    dQmax = 2e-3
    phi = np.linspace(0, 2*pi, 1000)
    sbs.set_palette('Blues')
    fig, ax = plt.subplots(1, 1)
    # ax.plot(phi, B(B_integrand, Jz=0, phi=phi) /
    #         dQmax, label='$J_z/\epsilon_z=0$')
    # ax.plot(phi, B(B_integrand, Jz=1, phi=phi) /
    #         dQmax,  label='$J_z /\epsilon_z=1$')
    # ax.plot(phi, B(B_integrand, Jz=2, phi=phi) /
    #         dQmax,  label='$J_z /\epsilon_z=2$')
    # ax.plot(phi, B(B_integrand, Jz=3, phi=phi) /
    #         dQmax,  label='$J_z /\epsilon_z=3$')
    # time_start = time.process_time()
    # p = 0
    # l = 0
    # print('Order of magnitude for longitudinal amplitude: ',
    #   SIGMA_Z/(BETA*c)*OMEGA_REV)

    # Hr, Hi = H(Jz, p=p, l=l)
    z_1 = 0
    z_2 = np.linspace(0, 2, 20)*ive(1, 1)
    Hres = H_sum(z_1, z_2, l=0, n_max=10)
    plt.plot(z_2, Hres.real, marker='.')
    plt.plot(z_2, Hres.imag, marker='.')

    # time_elapsed = time.process_time()-time_start
    # print('Time elapsed: {0:.2e}'.format(time_elapsed))
    # omega_p = p*OMEGA_REV+Q_X*OMEGA_REV+l*OMEGA_S
    ax.set_xlabel('p')
    ax.set_ylabel('$H$')
    # ax.set_ylabel(
    # '$B(J_z, \phi)$ [$\\frac{\Delta Q_\mathrm{max}}{Q_\mathrm{s}}$]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlim(0, 2*pi)
    # ticks = np.linspace(0, 2*pi, 5)
    # ax.set_xticks(ticks)
    # ax.minorticks_on()
    # ax.set_xticklabels(
    # ['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$'],)
    ax.xaxis.grid()
    plt.figlegend(frameon=False)
    plt.savefig('Results/'+'H.pdf', bbox_inches='tight')
    plt.show()
