from seaborn import palettes
from tune_calculation import *
if __name__ == '__main__':
    n_particles = int(1e5)
    Jx = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    Jy = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    dQxel, dQyel = get_elens_tune(
        dQmax=0.001, Jx=Jx, Jy=Jy, ratio=1., simplified=False)
    rms_x, rms_y = np.sqrt(np.var(dQxel)), np.sqrt(np.var(dQyel))
    palette = sbs.color_palette('RdBu')
    ax = plot_spread(dQxel, dQyel, normalise=True, filename=None)
    ax.ax_joint.plot(np.mean(dQxel)/rms_x, np.mean(dQyel)/rms_y,
                     marker='o', c='darkgreen', label='Lens coherent tune shift')
    ax.ax_joint.plot(0, 0, marker='o', c='darkgrey', label='Machine tune')

    ax.ax_joint.legend(loc='lower right', frameon=False)
    plt.savefig(
        'Results/'+'GS-elens'+'.pdf', bbox_inches='tight')
    plt.show()
