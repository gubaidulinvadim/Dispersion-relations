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
    ax = 0.92e-4
    ay = 0.96e-4
    bxy = 0.65e-4
    a = np.array(((ax, -bxy), (-bxy, ay)))
    @jit(nogil=True)
    def tune_dist_funcEL(J_x, J_y):
        return get_elens_tune_for_round_beam_simplified(1e-3, J_x, J_y)
    dispersion_solver = TransverseDispersionRelation(tune_dist_funcEL)
    Qs = 1.74e-3
    mode = 0
    tune_vec = np.linspace(-Qs, Qs, 100)
    real_vec, imag_vec = dispersion_solver.dispersion_relation(
        tune_vec, Qs, mode=mode)
    stab_vec_re, stab_vec_im = dispersion_solver.tune_shift(
        real_vec, imag_vec)
    folder = '/home/vgubaidulin/PhD/Data/DR/elens(m={0:})/'.format(mode)
    # save_results(folder, stab_vec_re, stab_vec_im, tune_vec)
    palette = sbs.color_palette('RdBu')
    col = palette[0]
    fig, ax = plt.subplots(1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
    ax.minorticks_on()
    ax.set_xlabel(
        '$\Re{\Delta Q_{\mathrm{coh}}}$ $[\delta Q_{\mathrm{s}}]$', size=30)
    ax.set_ylabel(
        '$\Im{\Delta Q_{\mathrm{coh}}}$ $[\delta Q_{\mathrm{s}}]$', size=30)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, .15)
    plt.plot(stab_vec_re/Qs, stab_vec_im/Qs, c=col)
    plt.savefig('Results/'+'GS_elens_DR.pdf', bbox_inches='tight')
    plt.show()
