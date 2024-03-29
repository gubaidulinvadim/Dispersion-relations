from dispersion_relation_calculation import *
import matplotlib.ticker as ticker
sbs.set(rc={'figure.figsize': (8.3, 5.2),
            'text.usetex': True,
            'font.family': 'serif',
            'font.size': 20,
            'axes.linewidth': 2,
            'lines.linewidth': 4,
            'legend.fontsize': 24,
            'legend.numpoints': 1, },
        style='ticks',
        palette='RdBu',
        context='talk',
        font_scale=1.4)
if __name__ == '__main__':
    @np.vectorize
    def tune_dist_funcPEL(J_z):
        return get_pelens_tune(J_z, max_tune_shift_x=1e-3, max_tune_shift_y=1e-3)
    dispersion_solver = LongitudinalDispersionRelation(tune_dist_funcPEL)
    Q_S = 1.74e-3
    mode = 0
    tune_vec = np.linspace(-(Q_S+np.abs(mode)*Q_S), np.abs(mode)*Q_S+Q_S, 500)
    real_vec, imag_vec = dispersion_solver.dispersion_relation(
        tune_vec, Q_S, mode=mode)
    stab_vec_re, stab_vec_im = dispersion_solver.tune_shift(
        real_vec, imag_vec)
    folder = '/home/vgubaidulin/PhD/Data/DR/pelens(m={0:})/'.format(mode)
    save_results(folder, stab_vec_re, stab_vec_im, tune_vec)
    palette = sbs.color_palette('RdBu')
    col = palette[-1]
    fig, ax = plt.subplots(1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
    ax.minorticks_on()
    ax.set_xlabel(
        '$\Re{\Delta Q_{\mathrm{coh}}}$ $[Q_{\mathrm{s}}]$', size=30)
    ax.set_ylabel(
        '$\Im{\Delta Q_{\mathrm{coh}}}$ $[Q_{\mathrm{s}}]$', size=30)
    ax.set_xlim(-.5, .5)
    ax.set_ylim(0, .15)
    plt.plot(stab_vec_re/Q_S, stab_vec_im/Q_S, c=col)
    plt.savefig(
        'Results/'+'LHC_PEL_DR(m={0:}).pdf'.format(mode), bbox_inches='tight')
    plt.show()
