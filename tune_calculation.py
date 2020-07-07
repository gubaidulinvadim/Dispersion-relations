import numpy as np
import scipy as sp
from scipy.constants import epsilon_0, c, m_e, m_p, e, pi
from scipy.special import i0, i1
from scipy.integrate import quad
from matplotlib import pyplot as plt
import seaborn as sbs
sbs.set(rc={'figure.figsize':(8.3,5.2)}, style='white', palette='colorblind', context='talk')

def get_octupole_tune(a, J):
    '''
    @params: 
    J -- vector of transverse action variables: [J_x, J_y];
    a -- martix of octupole coefficients: [[a_xx, a_xy],[a_yx, a_yy]]
    Returns vector of tune shifts [Q_x, Q_y] from octupoles for a given [J_x, J_y]
    '''
    dQ = np.dot(a, J)
    return dQ
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

if __name__ == '__main__':
    Ekin = 7e12
    epsn = 2.5e-6

    p0 = Ekin*e/c
    gamma = 1+Ekin*e/(m_p*c**2)
    a1 = epsn/gamma*get_octupole_coefficients(p0, 175.5, 33.6, 63100, 84, 0.32) 
    a2 = epsn/gamma*get_octupole_coefficients(p0, 30.1, 178.8, 63100, 84, 0.32)
    a = a1 + a2

    n_particles = 16384
    J =  np.random.exponential(1.0, size=(2, n_particles)) 
    # plt.scatter(J[0], J[1], marker='.')
    # plt.show()
    # dQ = get_octupole_tune(a, J)
    # dQrms_x = np.sqrt(np.var(dQ[0]))
    # dQrms_y = np.sqrt(np.var(dQ[1]))
    # # plt.scatter(dQ[0]/dQrms_x, dQ[1]/dQrms_y, marker='.')
    # ax = sbs.jointplot(dQ[0]/dQrms_x, dQ[1]/dQrms_y, kind='hex')

    Jx, Jy = J[0], J[1]
    dQx, dQy = get_elens_tune(0.001, Jx, Jy)
    dQrms_x, dQrms_y = np.sqrt(np.var(dQx)), np.sqrt(np.var(dQy))
    print('RMS tune spread obtained are: {0:.3f}, {1:.3f}'.format(dQrms_x, dQrms_y))
    ax = sbs.jointplot(dQx/dQrms_x, dQy/dQrms_y, kind='hex')
    ax.ax_joint.set_xlabel(r'$\Delta Q_x / \delta Q_{rms}$')
    ax.ax_joint.set_ylabel(r'$\Delta Q_x / \delta Q_{rms}$')
    ax.ax_joint.set_xlim(0, 6)
    ax.ax_joint.set_ylim(0, 6)
    ax.ax_joint.set_xticks(np.linspace(-0, 6, 5))
    ax.ax_joint.set_yticks(np.linspace(-0, 6, 5))
    plt.tight_layout()
    # plt.savefig('/home/vgubaidulin/PhD/Results/Tune_spreads/Electron_lens_matched_theory.png')
    # plt.savefig('/home/vgubaidulin/PhD/Results/Tune_spreads/Electron_lens_matched_theory.svg')
    # plt.savefig('/home/vgubaidulin/PhD/Results/Tune_spreads/Electron_lens_matched_theory.eps')
    plt.show()

