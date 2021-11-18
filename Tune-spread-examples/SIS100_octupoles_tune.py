import numpy as np
import scipy as sp
import scipy.stats as ss
from matplotlib import pyplot as plt
import seaborn as sbs
from tune_calculation import *


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
                       ratio=3)
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
    sbs.set(rc={'figure.figsize': (8.3, 5.2)}, style='ticks',
            palette='colorblind', context='talk')
    Ekin = 2e8
    n_particles = int(1e5)
    p0 = Ekin*e/c
    gamma = 1+Ekin*e/(m_p*c**2)
    beta = np.sqrt(1-gamma**-2)
    K3 = 50
    epsnx = 0.25*48e-6  # 0.25*35e-6
    epsny = 0.25*48e-6  # 0.25*15e-6
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
    plot_spread(dQx, dQy, normalise=False, filename=None)
    plt.savefig(
        'Results/'+'SIS100OCT_K3={0:}.pdf'.format(K3), bbox_inches='tight')
    plt.show()
