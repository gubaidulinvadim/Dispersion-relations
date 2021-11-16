from numpy.lib.type_check import real
from dispersion_relation_calculation import *
from LHC_constants import *
MAX_INTEGRAL_LIMIT = 16*SIGMA_Z
EPSILON = 1e-6
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
Q_X_FRAC = Q_X-np.floor(Q_X)
OMEGA_X = OMEGA_REV*Q_X
if __name__ == '__main__':
    n_particles = int(1e5)

    def func(r, mode):
        Jz = .5*(r/SIGMA_Z)**2
        a = OMEGA_X*SIGMA_Z/(BETA*c)
        return np.sqrt(2*Jz)*np.exp(-Jz)*jv(abs(mode), a*np.sqrt(2*Jz))**2
    print('Bessel function argument: {0:.2e}'.format(
        OMEGA_X/(BETA*c)*SIGMA_Z))
    mode = 0

    def normalisation(mode=0):
        return quad(func, 0, np.infty, args=(mode,))[0]
    N = normalisation(mode)
    print('Dispersion integral normalisation: {0:.2e}'.format(N))

    def tune_dist_funcPEL(r):
        dQmax = 1e-3
        Jz = .5*(r/SIGMA_Z)**2
        return get_pelens_tune(Jz, max_tune_shift_x=dQmax, max_tune_shift_y=dQmax)
    dispersion_solver = LongitudinalDispersionRelation2(
        tune_dist_funcPEL, beta=BETA, beta_z=BETA_Z, omega_betax=OMEGA_X, sigma_z=SIGMA_Z)
    dQmax = 1e-3
    tune_vec = np.linspace(-.25*dQmax, 1.25*dQmax, 10000)
    real_vec, imag_vec = dispersion_solver.dispersion_relation(
        tune_vec, Q_S, mode=mode)
    real_vec /= N
    imag_vec /= N
    np.save('/home/vgubaidulin/PhD/Data/real_vec.npy', real_vec)
    np.save('/home/vgubaidulin/PhD/Data/imag_vec.npy', imag_vec)

    stab_vec_re, stab_vec_im = dispersion_solver.tune_shift(
        real_vec, imag_vec)
    folder = '/home/vgubaidulin/PhD/Data/DR/PELSIS100/'.format(mode)
    save_results(folder, stab_vec_re, stab_vec_im, tune_vec)
    palette = sbs.color_palette('RdBu')
    col = palette[-1]
    fig, ax = plt.subplots(1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel(
        '$\Re{\Delta Q_{\mathrm{coh}}}$ $[Q_{\mathrm{s}}]$', size=30)
    ax.set_ylabel(
        '$\Im{\Delta Q_{\mathrm{coh}}}$ $[Q_{\mathrm{s}}]$', size=30)
    # ax.set_xlim(-3, 3)
    print(max(imag_vec))
    # ax.set_ylim(0, .15)
    # plt.plot(stab_vec_re/Qs, stab_vec_im/Qs, c=col)
    plt.plot(stab_vec_re/(dQmax), stab_vec_im/(dQmax), c=col)
    plt.savefig('Results/'+'SIS100_PEL_DR2.pdf', bbox_inches='tight')
    plt.show()
