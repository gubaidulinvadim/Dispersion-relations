from seaborn import palettes
from tune_calculation import *


def plot_EL():
    n_particles = int(1e5)
    Jx = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    Jy = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    dQxel, dQyel = get_elens_tune(
        dQmax=0.001, Jx=Jx, Jy=Jy, ratio=1., simplified=False)
    rms_x, rms_y = np.sqrt(np.var(dQxel)), np.sqrt(np.var(dQyel))
    palette = sbs.color_palette('RdBu')
    ax = plot_spread(dQxel, dQyel, normalise=True,
                     filename=None, xlim=(0, 7), ylim=(0, 7), color=palette[0])
    ax.ax_joint.plot(np.mean(dQxel)/rms_x, np.mean(dQyel)/rms_y,
                     marker='o', c='darkgreen', label='Lens coherent tune shift')
    ax.ax_joint.plot(0, 0, marker='o', c='darkgrey', label='Machine tune')
    ax.ax_joint.axvline(x=np.mean(dQxel)/rms_x,
                        c='darkgreen', linestyle='dashed', linewidth=1, alpha=0.85)
    ax.ax_joint.axhline(y=np.mean(dQyel)/rms_y,
                        c='darkgreen', linestyle='dashed', linewidth=1, alpha=0.85)
    ax.ax_marg_x.axvline(np.mean(dQxel)/rms_x,
                         c='darkgreen', linestyle='dashed', linewidth=1, alpha=0.85)
    ax.ax_marg_y.axhline(np.mean(dQyel)/rms_y,
                         c='darkgreen', linestyle='dashed', linewidth=1, alpha=0.85)

    ax.ax_joint.legend(loc='lower right', frameon=False)
    plt.savefig(
        'Results/'+'GS-elens'+'.pdf', bbox_inches='tight')
    return ax


if __name__ == '__main__':
    ax = plot_EL()
    plt.show()
