from tune_calculation import *


def plot_spread(dQx, dQy, normalise=True):
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
    ax = sbs.jointplot(x=dQx/dQrms_x, y=dQy/dQrms_y, kind='hex',
                       color=palette[-2], marginal_kws={'bins': 25, }, ratio=3, xlim=(-0.1, 7), ylim=(-7, .1))
    ax.ax_joint.plot(np.mean(dQxRFQ)/dQrms_x, np.mean(dQyRFQ)/dQrms_y,
                     marker='o', c='darkgreen', label='RFQ coherent tune shift')
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
    Jz = trunc_exp_rv(low=0, high=2.5**2, scale=1.0, size=n_particles)
    v2 = 2.23e9
    dQxRFQ, dQyRFQ = get_rfq_tune(Jz, v2)
    rms_x, rms_y = np.sqrt(np.var(dQxRFQ)), np.sqrt(np.var(dQyRFQ))
    print(rms_x/max(dQxRFQ), rms_y/max(dQxRFQ))
    ax = plot_spread(dQxRFQ, dQyRFQ, normalise=True)
    # plt.savefig(
    # 'Results/'+'LHC_RFQ'+'.pdf', bbox_inches='tight')
    plt.show()
