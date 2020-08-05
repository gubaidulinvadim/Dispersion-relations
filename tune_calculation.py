import numpy as np
import scipy as sp
from numba import jit
from scipy.constants import epsilon_0, c, m_e, m_p, e, pi
from scipy.special import i0, i1, iv
from scipy.integrate import quad
from matplotlib import pyplot as plt
import seaborn as sbs
sbs.set(rc={'figure.figsize':(8.3,5.2)}, style='white', palette='colorblind', context='talk')

@jit(nogil=True)
def get_octupole_tune(a, J):
    '''
    @params: 
    J -- vector of transverse action variables: [J_x, J_y];
    a -- martix of octupole coefficients: [[a_xx, a_xy],[a_yx, a_yy]]
    Returns vector of tune shifts [Q_x, Q_y] from octupoles for a given [J_x, J_y]
    '''
    dQ = np.dot(a, J)
    return dQ
@jit(nogil=True)
def get_octupole_coefficients(p0, beta_x, beta_y, O3, N_oct, L):
    '''
    @params:
    p0 -- Beam energy E=p0*c;
    beta_x -- horizontal Twiss beta-function;
    beta_y -- vertical Twiss beta-function;
    O3 -- octupole strength;
    N_oct -- number of octupoles;
    L -- octupole length
    Returns matrix of octupole coefficients [[a_xx, a_xy],[a_yx, a_yy]]
    '''
    a = ( 3*e/(8*pi*p0)*N_oct*O3*L ) * np.ones(shape=(2, 2), dtype=np.float64)
    a[0, 0] *= beta_x*beta_x
    a[0, 1] *= -2*beta_x*beta_y
    a[1, 0] *= -2*beta_y*beta_x
    a[1, 1] *= beta_y*beta_y
    return a
@np.vectorize
def get_elens_tune(dQmax, Jx, Jy):
    dQx = quad(lambda u: ( i0(Jx*u) - i1(Jx*u)) * i0(Jy*u) * np.exp(-(Jx + Jy) * u), 0, 1 )[0]
    dQy = quad(lambda u: ( i0(Jy*u) - i1(Jy*u)) * i0(Jx*u) * np.exp(-(Jx + Jy) * u), 0, 1 )[0]
    return dQmax*dQx, dQmax*dQy
def get_pelens_tune(Jz, max_tune_shift):
    dQx = np.exp(-Jz/4)*i0(Jz/4)
    dQy = dQx
    return max_tune_shift*dQx, max_tune_shift*dQy

def plot_spread(dQx, dQy, filename=None, normalise=True):
    dQrms_x, dQrms_y = (np.sqrt(np.var(dQx)), np.sqrt(np.var(dQy))) if normalise else (1, 1)
    # print('RMS tune spread obtained are: {0:.3f}, {1:.3f}'.format(dQrms_x, dQrms_y))
    ax = sbs.jointplot(dQx/dQrms_x, dQy/dQrms_y, kind='hex')
    ax.ax_joint.set_xlabel(r'$\Delta Q_x / \delta Q_{rms}$')
    ax.ax_joint.set_ylabel(r'$\Delta Q_x / \delta Q_{rms}$')
    delta = 0.0011
    ax.ax_joint.set_xlim(-delta, delta)
    ax.ax_joint.set_ylim(-delta, delta)
    ax.ax_joint.set_xticks(np.linspace(-delta, delta, 5))
    ax.ax_joint.set_yticks(np.linspace(-delta, delta, 5))
    plt.tight_layout()
    if filename != None:
        plt.savefig('/home/vgubaidulin/PhD/Results/Tune_spreads/'+filename+'.png')
        plt.savefig('/home/vgubaidulin/PhD/Results/Tune_spreads/'+filename+'.svg')
        plt.savefig('/home/vgubaidulin/PhD/Results/Tune_spreads/'+filename+'.eps')
    plt.show()
if __name__ == '__main__':
    Ekin = 7e12
    epsn = 2.5e-6

    # p0 = Ekin*e/c
    # gamma = 1+Ekin*e/(m_p*c**2)
    # a1 = epsn/gamma*get_octupole_coefficients(p0, 140/72*175.5, 140/72*33.6, 63100, 25*84, 0.32) 
    # a2 = epsn/gamma*get_octupole_coefficients(p0, 140/72*30.1, 140/72*178.8, 63100, 25*84, 0.32)
    # a = -a1 - a2

    n_particles = 16384
    # J =  np.random.exponential(1.0, size=(2, n_particles)) 
    # # plt.scatter(J[0], J[1], marker='.')
    # # plt.show()
    # dQ = get_octupole_tune(a, J)
    # dQrms_x = np.sqrt(np.var(dQ[0]))
    # dQrms_y = np.sqrt(np.var(dQ[1]))

    # Jx, Jy = J[0], J[1]
    # dQx, dQy = get_elens_tune(0.001, Jx, Jy)
    # # plot_spread(dQx, dQy)

    Jz = np.random.exponential(1.0, n_particles)
    dQx, dQy = get_pelens_tune(Jz, 0.001)
    plot_spread(dQx, dQy, normalise=False)

