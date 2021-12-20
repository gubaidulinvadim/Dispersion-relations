from parameters.LHC_constants import SIGMA_Z
from tune_calculation import *


def plot_spread(dQx, dQy, normalise=True, color=None):
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
    ax = sbs.jointplot(x=dQx/dQrms_x, y=dQy/dQrms_y, kind='hex', color=color, marginal_kws={
                       'bins': 25, }, ratio=3,
                       # xlim=(-0.1, 7.), ylim=(-.1, 7.)
                       )
    ax.ax_joint.plot(np.mean(dQx)/dQrms_x, np.mean(dQy)/dQrms_y,
                     marker='o', c='darkgreen', label='PEL coherent tune shift')
    ax.ax_joint.plot(0, 0, marker='o', c='darkgrey', label='Machine tune')
    ax.ax_joint.set_xlabel(r'$\Delta Q_x/\Delta Q_\mathrm{RMS}$')
    ax.ax_joint.set_ylabel(r'$\Delta Q_y/\Delta Q_\mathrm{RMS}$')
    ax.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
    ax.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
    ax.ax_joint.minorticks_on()
    ax.ax_joint.legend(loc='lower right', frameon=False)
    return ax


if __name__ == '__main__':
    n_particles = int(1e7)
    palette = sbs.color_palette('RdBu')
    Jx = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    Jy = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    J = np.column_stack((Jx, Jy)).T
    a = -np.array([[9.20e-05, -6.54e-05], [-6.54e-05, 9.63e-05]])
    Jz = trunc_exp_rv(low=0, high=2.5**2, scale=1.0, size=n_particles)
    dQmax = 1e-3
    r = 0.5
    dQxPEL, dQyPEL = get_pelens_tune(
        Jz, max_tune_shift_x=dQmax*r, max_tune_shift_y=dQmax*r)
    dQxOCT, dQyOCT = get_elens_tune(
        dQmax=dQmax*(1-r), Jx=Jx, Jy=Jy, ratio=1., simplified=False)
    rms_x, rms_y = np.sqrt(np.var(dQxPEL+dQxOCT)
                           ), np.sqrt(np.var(dQyPEL+dQxOCT))
    ax = plot_spread(dQxPEL+dQxOCT, dQyPEL+dQyOCT,
                     normalise=False, color='#4CB391')
    ax.ax_marg_x.hist(dQxPEL,
                      color=palette[-1], alpha=0.5, bins=25)
    ax.ax_marg_y.hist(dQyPEL,
                      color=palette[-1], alpha=0.5, bins=25, orientation='horizontal')
    plt.savefig(
        'Results/'+'PELDCEL-'+'.pdf', bbox_inches='tight')
    plt.show()
