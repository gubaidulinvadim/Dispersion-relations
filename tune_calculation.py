import numpy as np
import scipy.stats as ss
from numba import jit
from scipy.constants import pi, c, m_p, e
from scipy.special import i0, i1, j0
from scipy.integrate import quad
from matplotlib import pyplot as plt
import seaborn as sbs

import matplotlib.ticker as ticker


def trunc_exp_rv(low, high, scale, size):
    rnd_cdf = np.random.uniform(ss.expon.cdf(x=low, scale=scale),
                                ss.expon.cdf(x=high, scale=scale),
                                size=size)
    return ss.expon.ppf(q=rnd_cdf, scale=scale)


def get_octupole_coefficients(beta_x, beta_y, K3, N_oct, L):
    '''
    @params:
    beta_x -- horizontal Twiss beta-function;
    beta_y -- vertical Twiss beta-function;
    O3 -- octupole strength;
    N_oct -- number of octupoles;
    L -- octupole length
    Returns matrix of octupole coefficients [[a_xx, a_xy],[a_yx, a_yy]]
    '''
    a = (3/(8*pi)*N_oct*K3/6*L) * np.ones(shape=(2, 2), dtype=np.float64)
    a[0, 0] *= beta_x*beta_x
    a[0, 1] *= -2*beta_x*beta_y
    a[1, 0] *= -2*beta_y*beta_x
    a[1, 1] *= beta_y*beta_y
    return a


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


# @jit(nogil=True)
@np.vectorize
def get_sc_tune_for_round_beam(dQmax, Jx, Jy):
    ax = np.sqrt(2.0*Jx)
    ay = np.sqrt(2.0*Jy)
    dQx = (192.0-11.0*ax-18.0*np.sqrt(ax*ay)+3.0*ax**2) / \
        (192.0-11.0*ax-18.0*np.sqrt(ax*ay)+3.0*ax**2+36.0*ax**2+21.0*ay**2)
    dQy = (192.0-11.0*ay-18.0*np.sqrt(ax*ay)+3.0*ay**2) / \
        (192.0-11.0*ay-18.0*np.sqrt(ax*ay)+3.0*ay**2+36.0*ay**2+21.0*ax**2)
    return -dQmax*dQx, -dQmax*dQy


@jit(nogil=True)
def get_elens_tune_for_round_beam_simplified(dQmax, Jx, Jy, ratio=1.0):
    ax = np.sqrt(2.0*Jx)/ratio
    ay = np.sqrt(2.0*Jy)/ratio
    dQx = (192.0-11.0*ax-18.0*np.sqrt(ax*ay)+3.0*ax**2) / \
        (192.0-11.0*ax-18.0*np.sqrt(ax*ay)+3.0*ax**2+36.0*ax**2+21.0*ay**2)
    dQy = (192.0-11.0*ay-18.0*np.sqrt(ax*ay)+3.0*ay**2) / \
        (192.0-11.0*ay-18.0*np.sqrt(ax*ay)+3.0*ay**2+36.0*ay**2+21.0*ax**2)
    return dQmax*dQx, dQmax*dQy


@np.vectorize
def get_elens_tune(dQmax: float, Jx: float, Jy: float, ratio: float = 1.0, simplified: bool = True):
    '''
    @params
    dQmax -- (float) Maximal tune shift at the beam's core
    Jx -- (float) horizontal action variable (normalized to the beam emittance)
    Jy -- (float) vertical action variable (normalized to the beam emittance)
    ratio -- (float) electron to ion beam profile ratio 
    simplified -- (bool) if True, uses numerical approximation for faster calculation time
    '''
    Jx /= ratio**2
    Jy /= ratio**2
    if simplified:
        return get_elens_tune_for_round_beam_simplified(dQmax, Jx, Jy, ratio=ratio)
    else:
        dQx = quad(lambda u: (i0(.5*Jx*u) - i1(.5*Jx*u)) * i0(.5*Jy*u)
                   * np.exp(-(.5*Jx + .5*Jy) * u), 0, 1)[0]
        dQy = quad(lambda u: (i0(.5*Jy*u) - i1(.5*Jy*u)) * i0(.5*Jx*u)
                   * np.exp(-(.5*Jx + .5*Jy) * u), 0, 1)[0]
    return dQmax*dQx, dQmax*dQy


@np.vectorize
def get_sc_tune_for_round_beam(dQmax, Jx, Jy, Jz, ratio=1.0, simplified=True):
    '''
    @params
    dQmax -- (float) Maximal tune shift at the beam's core
    Jx -- (float) horizontal action variable (normalized to the beam emittance)
    Jy -- (float) vertical action variable (normalized to the beam emittance)
    ratio -- (float) electron to ion beam profile ratio 
    simplified -- (bool) if True, uses numerical approximation for faster calculation time
    '''
    Jx /= 2*ratio**2
    Jy /= 2*ratio**2
    if simplified:
        return get_elens_tune_for_round_beam_simplified(dQmax, Jx, Jy)
    else:
        dQx = quad(lambda u: (i0(Jx*u) - i1(Jx*u)) * i0(Jy*u)
                   * np.exp(-(Jx + Jy) * u), 0, 1)[0]
        dQy = quad(lambda u: (i0(Jy*u) - i1(Jy*u)) * i0(Jx*u)
                   * np.exp(-(Jx + Jy) * u), 0, 1)[0]
    return dQmax*dQx*i0(-Jz/4)*np.exp(-Jz/4), dQmax*dQy*i0(-Jz/4)*np.exp(-Jz/4)


@np.vectorize
def get_chroma_tune(dp, Qpx, Qpy, Ekin, A=40):
    gamma = 1+Ekin*e/(m_p*c**2)
    p0 = np.sqrt(gamma**2 - 1) * A * m_p * c
    return dp*Qpx/p0, dp*Qpy/p0


def get_pelens_tune(Jz, max_tune_shift_x=1e-3, max_tune_shift_y=1e-3):
    '''
    @params 
    Jz -- longitudinal action (normalized to the longitudinal beam emittance)
    max_tune_shift -- maximal tune shift from pulsed electron lens                                                                                                                                                                                                                 
    '''
    dQx = i0(-.25*Jz)*np.exp(-.25*Jz)
    return max_tune_shift_x*dQx, max_tune_shift_y*dQx


def get_rfq_tune(Jz, v_2):
    sigma_z, omegaRFQ, beta_z = 0.06, 800e6*2*pi, 815.6
    Jz *= sigma_z**2/(2*beta_z)
    p0 = 6.5e3
    beta = 1
    beta_x, beta_y = 92.7, 93.2
    dQmax_x = +beta_x/(2*pi)*v_2/(omegaRFQ*p0)
    dQmax_y = -beta_y/(2*pi)*v_2/(omegaRFQ*p0)
    dQ = j0(omegaRFQ/(beta*c)*np.sqrt(2*Jz*beta_z))
    return dQmax_x*dQ, dQmax_y*dQ


def plot_spread(dQx, dQy, filename=None, normalise=True):
    sbs.set(rc={'figure.figsize': (8.3, 5.2),
                'text.usetex': True,
                'font.family': 'arial',
                'font.size': 20,
                'axes.linewidth': 2,
                'lines.linewidth': 3,
                'legend.fontsize': 16,
                'legend.numpoints': 1, },
            style='ticks',
            palette='RdBu',
            context='talk')
    dQrms_x, dQrms_y = (np.sqrt(np.var(dQx)), np.sqrt(
        np.var(dQy))) if normalise else (1., 1.)
    palette = sbs.color_palette('RdBu')
    ax = sbs.jointplot(x=dQx/dQrms_x, y=dQy/dQrms_y, kind='hex', color=palette[1], marginal_kws={'bins': 25,
                                                                                                 # 'hist': True,
                                                                                                 # 'hist_kws': {'density': True}
                                                                                                 },
                       ratio=3, xlim=(-5, 5), ylim=(-5, 5))
    # / \delta Q_{rms}$')
    ax.ax_joint.set_xlabel(r'$\Delta Q_x/\Delta Q_\mathrm{RMS}$')
    # / \delta Q_{rms}$')
    ax.ax_joint.set_ylabel(r'$\Delta Q_y/\Delta Q_\mathrm{RMS}$')
    ax.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
    ax.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
    ax.ax_joint.minorticks_on()
    if filename != None:
        plt.savefig(
            '/home/vgubaidulin/PhD/Results/Tune_spreads/'+filename+'.png', bbox_inches='tight')
        plt.savefig(
            '/home/vgubaidulin/PhD/Results/Tune_spreads/'+filename+'.pdf', bbox_inches='tight')
    # plt.show()
    return ax


if __name__ == '__main__':
    # Ekin = 2e8  # 6.5e12
    n_particles = int(2e5)
    # dQrms_oct_x = np.empty(shape=(5,), dtype=np.float64)
    # dQrms_elens_x = np.empty(shape=(5,), dtype=np.float64)
    # dQrms_combo_x = np.empty(shape=(5,), dtype=np.float64)
    # Jz = trunc_exp_rv(0, 2.5**2
    # , 1.0, n_particles)
    # p0 = Ekin*e/c
    # gamma = 1+Ekin*e/(m_p*c**2)
    # beta = np.sqrt(1-gamma**-2)
    # epsx = 0.25*35e-6
    # epsy = 0.25*15e-6
    # def get_detuning_from_SIS100_octupoles(Ekin, K3):
    #     beta_xF, beta_xD = 5.8, 17.2
    #     beta_yF, beta_yD = 17.7, 5.8
    #     Lm = 0.75
    #     Nf = 6
    #     Nd = -6
    #     A, Q = 238, 28
    #     Brho = 3.3357*A/Q*Ekin/1e9
    #     print(Brho)
    #     axx = 3/(8*np.pi)*(e/c*Ekin)*Lm*(Nf*beta_xF**2*K3/6+2*Nd*beta_xD**2*K3/6)/(e/c*Ekin)
    #     axy = 3/(8*np.pi)*(e/c*Ekin)*Lm*(2*Nf*beta_xF*beta_yF*K3/6+2*Nd*beta_xD*beta_yD*K3/6)/(e/c*Ekin)
    #     ayx = 3/(8*np.pi)*(e/c*Ekin)*Lm*(2*Nf*beta_xF*beta_yF*K3/6+2*Nd*beta_xD*beta_yD*K3/6)/(e/c*Ekin)
    #     ayy = 3/(8*np.pi)*(e/c*Ekin)*Lm*(Nf*beta_yF**2*K3/6+2*Nd*beta_yD**2*K3/6)/(e/c*Ekin)
    #     return ((axx, axy), (ayx, ayy))
    # K3 = 50
    # Ekin = 2e8

    # gamma = 1 + Ekin * e / (m_p * c**2)
    # a1 = (get_detuning_from_SIS100_octupoles(Ekin, K3))
    # b = np.array((epsx/gamma, epsy/gamma))
    # a = a1*b
    # J = np.random.exponential(1.0, size=(2, n_particles))
    Jx = trunc_exp_rv(0, 4, 1.0, n_particles)
    Jy = trunc_exp_rv(0, 4, 1.0, n_particles)
    # A = 40
    # Ekin = 8.576e6
    # gamma = 1 + Ekin * e / (m_p * c**2)
    # p0 = np.sqrt(gamma**2 - 1) * A * m_p * c

    # dpp = np.random.normal(p0, .5e-3*p0, 16384)-p0
    dQxel, dQyel = get_elens_tune(
        0.001, Jx, Jy, ratio=.5, simplified=False)
    # dQxel, dQyel = get_pelens_tune(Jz, max_tune_shift=1e-3)
    # dQxsc, dQysc = get_sc_tune_for_round_beam(-0, Jx, Jy, Jz)
    # dQxel, dQyel = get_rfq_tune(Jz, 2.23e9)
    # np.save('/home/vgubaidulin/PhD/Data/tmp/dQELx.npy', dQxel)
    # np.save('/home/vgubaidulin/PhD/Data/tmp/dQELy.npy', dQyel)
    # print('Total elens tune spread: {0:.5f}'.format(max(dQxel)-min(dQxel)))
    print('Maximal elens tune shift: {0:.3e} in x, {1:.3e} in y'.format(
        max(dQxel), max(dQyel)))
    # print('RMS elens tune spread: {0:.2f} [$\Delta Q$] in x, {1:.2f} [\Delta Q] in y'.format(np.sqrt(np.var(dQxel))/max(dQxel), np.sqrt(np.var(dQyel))/max(dQyel)))
    rms_x = np.sqrt(np.var(dQxel))
    rms_y = np.sqrt(np.var(dQyel))

    # print(min(dQxsc), max(dQxel))
    ax = plot_spread(dQxel, dQyel, normalise=True, filename=None)
    ax.ax_joint.plot(np.mean(dQxel)/rms_x, np.mean(dQyel) /
                     rms_y, marker='o', c='r')
    # print(np.mean(dQxel)/max(dQxel), np.mean(dQyel)/max(dQyel))
    # plt.savefig('/home/vgubaidulin/PhD/pelens_sc0.pdf', bbox_inches = 'tight')
    plt.show()
