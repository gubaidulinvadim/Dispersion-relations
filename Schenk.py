from parameters.LHC_constants import *
import numpy as np
import scipy as sp
from scipy.constants import epsilon_0, c, m_e, m_p, e, pi
from scipy.special import i0, i1, iv, j0, j1, jv
from scipy.integrate import quad
from matplotlib import pyplot as plt
import seaborn as sbs
import time
from tune_calculation import *
# MAX_INTEGRAL_LIMIT = 16
# EPSILON = 1e-6


@np.vectorize
def Q_detuning(phi: float, Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.5*Jz*(1-np.cos(2*phi)))


@np.vectorize
def Q_average_detuning(Jz: float, dQmax=0.001):
    return dQmax*np.exp(-0.5*Jz)*i0(0.5*Jz)


@np.vectorize
def B_integrand(phi: float, Jz: float, dQmax=.002):
    return Q_detuning(phi, Jz, dQmax) - Q_average_detuning(Jz, dQmax)


@np.vectorize
def Q_detuning_RFQ(phi: float, Jz: float, dQmax=0.001):
    return dQmax*np.cos(np.sqrt(2*Jz)*np.sin(phi))


@np.vectorize
def Q_average_detuning_RFQ(Jz: float, dQmax=0.001):
    return dQmax*j0(np.sqrt(2*Jz))


@np.vectorize
def B_integrand_RFQ(phi: float, Jz: float, dQmax=.002):
    return Q_detuning_RFQ(phi, Jz, dQmax) - Q_average_detuning_RFQ(Jz, dQmax)


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
    ax.plot(phi, B(B_integrand, Jz=0, phi=phi) /
            dQmax, label='$J_z/\epsilon_z=0$')
    ax.plot(phi, B(B_integrand, Jz=1, phi=phi) /
            dQmax,  label='$J_z /\epsilon_z=1$')
    ax.plot(phi, B(B_integrand, Jz=2, phi=phi) /
            dQmax,  label='$J_z /\epsilon_z=2$')
    ax.plot(phi, B(B_integrand, Jz=3, phi=phi) /
            dQmax,  label='$J_z /\epsilon_z=3$')
    # print('max value of the B function:',
    #       max(B(B_integrand, 3, phi)/dQmax))
    time_start = time.process_time()
    # p = 0
    # l = 0
    # print('Order of magnitude for longitudinal amplitude: ',
    #   SIGMA_Z/(BETA*c)*OMEGA_REV)

    # Hr, Hi = H(Jz, p=p, l=l)
    # time_elapsed = time.process_time()-time_start
    # print('Time elapsed: {0:.2e}'.format(time_elapsed))
    # omega_p = p*OMEGA_REV+Q_X*OMEGA_REV+l*OMEGA_S
    ax.set_xlabel('$\phi_z$')
    ax.set_ylabel(
        '$B(J_z, \phi)$ [$\\frac{\Delta Q_\mathrm{max}}{Q_\mathrm{s}}$]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 2*pi)
    ticks = np.linspace(0, 2*pi, 5)
    ax.set_xticks(ticks)
    ax.minorticks_on()
    ax.set_xticklabels(
        ['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$'],)
    ax.xaxis.grid()
    plt.figlegend(frameon=False)
    plt.savefig('Results/'+'B.pdf', bbox_inches='tight')
    plt.show()
