import numpy as np
from scipy.constants import c, m_p, e, pi
from scipy.integrate import quad, dblquad, tplquad
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sbs
from scipy.special import jv
from parameters.SIS100_constants import *
from tune_calculation import *
MAX_INTEGRAL_LIMIT = 16
EPSILON = 1e-6


def complexquad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))

    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0]+1j*imag_integral[0]


def complexdblquad(func, a, b, g, h, **kwargs):
    def real_func(x, **kwargs):
        return np.real(func(x, **kwargs))

    def imag_func(x, **kwargs):
        return np.imag(func(x, **kwargs))
    real_integral = dblquad(real_func, a, b, g, h, **kwargs)
    imag_integral = dblquad(imag_func, a, b, g, h, **kwargs)
    return real_integral[0]+1j*imag_integral[0]


class TransverseDispersionRelation():
    def __init__(self, tune_distribution_function):
        self.function = tune_distribution_function

    def distribution_func(self, J_x, J_y):
        return -np.exp(-J_x-J_y)

    def dispersion_integrand(self, J_x, J_y, tune: float, Qs, mode=0):
        '''
        J_x: horizontal action variable normalized to emittance
        J_y: vertical action variable normalized to emittance
        tune: dimensionless tune corresponding to frequency of external excitation
        '''
        tune_shift_x, tune_shift_y = self.function(J_x, J_y)
        return -self.distribution_func(J_x, J_y)*J_x/(tune - tune_shift_x - mode*Qs + 1j*EPSILON)

    def real_part_of_integrand(self, *args):
        return np.real(self.dispersion_integrand(*args))

    def imaginary_part_of_integrand(self, *args):
        return np.imag(self.dispersion_integrand(*args))

    def compute_real_part(self, tune, Qs, mode=0):
        r = dblquad(self.real_part_of_integrand, 0., MAX_INTEGRAL_LIMIT,
                    lambda x: 0., lambda x: MAX_INTEGRAL_LIMIT, args=(tune, Qs, mode))[0]
        return r

    def compute_imag_part(self, tune, Qs, mode=0):
        i = dblquad(self.imaginary_part_of_integrand, 0., MAX_INTEGRAL_LIMIT,
                    lambda x: 0., lambda x: MAX_INTEGRAL_LIMIT, args=(tune, Qs, mode))[0]
        return i

    def dispersion_relation(self, tune_vec, Qs, mode=0):
        real_vec = np.empty(shape=(len(tune_vec),), dtype=np.float64)
        imag_vec = np.empty(shape=(len(tune_vec), ), dtype=np.float64)
        for (j, tune) in tqdm(enumerate(tune_vec), total=len(tune_vec)):
            real_vec[j] = self.compute_real_part(tune, Qs, mode)
            imag_vec[j] = self.compute_imag_part(tune, Qs, mode)
        return real_vec, imag_vec

    def dispersion_relation2(self, tune, Qs, mode=0):
        vec = complexdblquad(self.dispersion_integrand, 0.,
                             MAX_INTEGRAL_LIMIT, lambda x: 0., lambda x: MAX_INTEGRAL_LIMIT, args=(tune, Qs, mode))
        stab_vec = np.divide(vec, np.absolute(vec))
        return stab_vec

    def tune_shift(self, real_vec, imag_vec):
        ampl_vec = real_vec*real_vec+imag_vec*imag_vec
        stab_vec_re = np.divide(real_vec, ampl_vec)
        stab_vec_im = np.divide(-imag_vec, ampl_vec)
        return stab_vec_re, stab_vec_im


class LongitudinalDispersionRelation(TransverseDispersionRelation):
    def __init__(self, tune_distribution_function):
        super().__init__(tune_distribution_function)

    def distribution_func(self, Jz):
        return np.exp(-Jz)

    def dispersion_integrand(self, Jz, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jz)
        return self.distribution_func(Jz)*Jz**(np.sqrt(mode**2))/(tune-tune_x-mode*Qs+1j*EPSILON)

    def compute_real_part(self, tune, Qs, mode=0):
        r = quad(self.real_part_of_integrand, 0.,
                 MAX_INTEGRAL_LIMIT, args=(tune, Qs, mode))[0]
        return r

    def compute_imag_part(self, tune, Qs, mode=0):
        i = quad(self.imaginary_part_of_integrand, 0.,
                 MAX_INTEGRAL_LIMIT, args=(tune, Qs, mode))[0]
        return i


class LongitudinalDispersionRelation2(LongitudinalDispersionRelation):
    def __init__(self, tune_distribution_function, sigma_z, beta_z, omega_betax, beta):
        self.sigma_z = sigma_z
        self.beta_z = beta_z
        self.omega_betax = omega_betax
        self.beta = beta
        self.a = self.omega_betax*self.sigma_z/(self.beta*c)
        super().__init__(tune_distribution_function)

    def distribution_func(self, r):
        Jz = .5*(r/self.sigma_z)**2
        return np.exp(-Jz)

    def dispersion_integrand(self, r, tune, Qs, mode=0):
        tune_x, tune_y = self.function(r)
        Jz = .5*(r/self.sigma_z)**2
        Bessel_func = jv(abs(mode), self.a*np.sqrt(2*Jz))
        return np.sqrt(2*Jz)*self.distribution_func(r)*Bessel_func**2/(tune-tune_x-mode*Qs+1j*EPSILON)


class TransverseDispersionRelationWithSpaceCharge(TransverseDispersionRelation):
    def __init__(self, tune_distribution_function, dQmax_sc):
        super().__init__(tune_distribution_function)
        self.dQmax_sc = dQmax_sc
        self.sc_function = get_elens_tune_for_round_beam_simplified

    def dispersion_integrand(self, Jx, Jy, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jx, Jy)
        sc_tune_x, sc_tune_y = self.sc_function(-self.dQmax_sc, Jx, Jy)
        return -self.distribution_func(Jx, Jy)/(tune-tune_x-sc_tune_x-mode*Qs+1j*EPSILON)

    def sc_dispersion_integrand(self, Jx, Jy, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jx, Jy)
        sc_tune_x, sc_tune_y = self.sc_function(-self.dQmax_sc, Jx, Jy)
        return -sc_tune_x*self.distribution_func(Jx, Jy)/(tune-tune_x-sc_tune_x-mode*Qs+1j*EPSILON)

    def sc_real_part_of_integrand(self, *args):
        return np.real(self.sc_dispersion_integrand(*args))

    def sc_imag_part_of_integrand(self, *args):
        return np.imag(self.sc_dispersion_integrand(*args))

    def sc_compute_real_part(self, tune, Qs):
        r = dblquad(self.sc_real_part_of_integrand, 0,
                    MAX_INTEGRAL_LIMIT, lambda x: 0, lambda x: MAX_INTEGRAL_LIMIT, args=(tune, Qs))[0]
        return r

    def sc_compute_imag_part(self, tune, Qs):
        i = dblquad(self.sc_imag_part_of_integrand, 0,
                    MAX_INTEGRAL_LIMIT, lambda x: 0, lambda x: MAX_INTEGRAL_LIMIT, args=(tune, Qs))[0]
        return i

    def sc_component_dispersion_relation(self, tune, Qs):
        sc_real_vec = np.empty(shape=(len(tune_vec),), dtype=np.float64)
        sc_imag_vec = np.empty(shape=(len(tune_vec), ), dtype=np.float64)
        for (j, tune) in tqdm(enumerate(tune_vec), total=len(tune_vec)):
            sc_real_vec[j] = self.sc_compute_real_part(tune, Qs)
            sc_imag_vec[j] = self.sc_compute_imag_part(tune, Qs)
        return sc_real_vec, sc_imag_vec

    def tune_shift(self, real_vec, imag_vec, sc_real_vec, sc_imag_vec):
        ampl_vec = real_vec*real_vec+imag_vec*imag_vec
        stab_vec_re = np.divide(
            real_vec+real_vec*sc_real_vec+imag_vec*sc_imag_vec, ampl_vec)
        stab_vec_im = np.divide(-imag_vec-imag_vec *
                                sc_real_vec+real_vec*sc_imag_vec, ampl_vec)
        return stab_vec_re, stab_vec_im


class LongitudinalDispersionRelationWithSpaceCharge(LongitudinalDispersionRelation):
    def __init__(self, tune_distribution_function, dQmax_sc):
        super().__init__(tune_distribution_function)
        self.dQmax_sc = dQmax_sc
        self.sc_function = get_pelens_tune

    def dispersion_integrand(self, Jz, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jz)
        sc_tune_x, sc_tune_y = self.sc_function(Jz, -self.dQmax_sc)
        return self.distribution_func(Jz)*Jz**(mode)/(tune-tune_x-sc_tune_x-mode*Qs+1j*EPSILON)

    def sc_dispersion_integrand(self, Jz, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jz)
        sc_tune_x, sc_tune_y = self.sc_function(Jz, -self.dQmax_sc)
        return sc_tune_x*self.distribution_func(Jz)*Jz**(mode)/(tune-tune_x-sc_tune_x-mode*Qs+1j*EPSILON)

    def sc_real_part_of_integrand(self, *args):
        return np.real(self.sc_dispersion_integrand(*args))

    def sc_imag_part_of_integrand(self, *args):
        return np.imag(self.sc_dispersion_integrand(*args))

    def sc_compute_real_part(self, tune, Qs):
        r = quad(self.sc_real_part_of_integrand, 0,
                 MAX_INTEGRAL_LIMIT, args=(tune, Qs))[0]
        return r

    def sc_compute_imag_part(self, tune, Qs):
        i = quad(self.sc_imag_part_of_integrand, 0,
                 MAX_INTEGRAL_LIMIT, args=(tune, Qs))[0]
        return i

    def sc_component_dispersion_relation(self, tune, Qs):
        sc_real_vec = np.empty(shape=(len(tune_vec),), dtype=np.float64)
        sc_imag_vec = np.empty(shape=(len(tune_vec), ), dtype=np.float64)
        for (j, tune) in tqdm(enumerate(tune_vec), total=len(tune_vec)):
            sc_real_vec[j] = self.sc_compute_real_part(tune, Qs)
            sc_imag_vec[j] = self.sc_compute_imag_part(tune, Qs)
        return sc_real_vec, sc_imag_vec

    def tune_shift(self, real_vec, imag_vec, sc_real_vec, sc_imag_vec):
        ampl_vec = real_vec*real_vec+imag_vec*imag_vec
        stab_vec_re = np.divide(
            real_vec+real_vec*sc_real_vec+imag_vec*sc_imag_vec, ampl_vec)
        stab_vec_im = np.divide(-imag_vec-imag_vec *
                                sc_real_vec+real_vec*sc_imag_vec, ampl_vec)
        return stab_vec_re, stab_vec_im


class GeneralizedDispersionRelation(TransverseDispersionRelation):
    def __init__(self, transverse_tune_distribution_function, longitudinal_tune_distribution_function):
        self.transverse_function = transverse_tune_distribution_function
        self.longitudinal_function = longitudinal_tune_distribution_function

    def distribution_func(self, J_x, J_y, J_z):
        return -np.exp(-J_x-J_y-J_z)

    def dispersion_integrand(self, J_x, J_y, J_z, tune: float, Qs, mode=0):
        '''
        J_x: horizontal action variable normalized to emittance
        J_y: vertical action variable normalized to emittance
        tune: dimensionless tune corresponding to frequency of external excitation
        '''
        transverse_tune_shift_x, transverse_tune_shift_y = self.transverse_function(
            J_x, J_y)
        longitudinal_tune_shift_x, longitudinal_tune_shift_y = self.longitudinal_function(
            J_z)

        return -self.distribution_func(J_x, J_y, J_z)*J_x*np.power(J_z, np.abs(mode))/(tune - transverse_tune_shift_x-longitudinal_tune_shift_x - mode*Qs + 1j*EPSILON)

    def real_part_of_integrand(self, *args):
        return np.real(self.dispersion_integrand(*args))

    def imaginary_part_of_integrand(self, *args):
        return np.imag(self.dispersion_integrand(*args))

    def compute_real_part(self, tune, Qs, mode=0):
        r = tplquad(self.real_part_of_integrand,
                    0., MAX_INTEGRAL_LIMIT,
                    lambda x: 0., lambda x: MAX_INTEGRAL_LIMIT,
                    lambda x, y: 0, lambda x, y: MAX_INTEGRAL_LIMIT,
                    args=(tune, Qs, mode))[0]
        return r

    def compute_imag_part(self, tune, Qs, mode=0):
        i = tplquad(self.imaginary_part_of_integrand,
                    0., MAX_INTEGRAL_LIMIT,
                    lambda x: 0., lambda x: MAX_INTEGRAL_LIMIT,
                    lambda x, y: 0, lambda x, y: MAX_INTEGRAL_LIMIT,
                    args=(tune, Qs, mode))[0]
        return i

    def dispersion_relation(self, tune_vec, Qs, mode=0):
        real_vec = np.empty(shape=(len(tune_vec),), dtype=np.float64)
        imag_vec = np.empty(shape=(len(tune_vec), ), dtype=np.float64)
        for (j, tune) in tqdm(enumerate(tune_vec), total=len(tune_vec)):
            real_vec[j] = self.compute_real_part(tune, Qs, mode)
            imag_vec[j] = self.compute_imag_part(tune, Qs, mode)
        return real_vec, imag_vec

    def dispersion_relation2(self, tune, Qs, mode=0):
        vec = complexdblquad(self.dispersion_integrand, 0.,
                             MAX_INTEGRAL_LIMIT, lambda x: 0., lambda x: MAX_INTEGRAL_LIMIT, args=(tune, Qs, mode))
        stab_vec = np.divide(vec, np.absolute(vec))
        return stab_vec

    def tune_shift(self, real_vec, imag_vec):
        ampl_vec = real_vec*real_vec+imag_vec*imag_vec
        stab_vec_re = np.divide(real_vec, ampl_vec)
        stab_vec_im = np.divide(-imag_vec, ampl_vec)
        return stab_vec_re, stab_vec_im


def save_results(folder, stab_vec_re, stab_vec_im, tune_vec):
    np.save(
        folder+'dQre.npy', stab_vec_re)
    np.save(
        folder+'dQim.npy', stab_vec_im)
    np.save(
        folder+'tunes.npy', tune_vec)


if __name__ == '__main__':
    Ekin = 7e12
    epsn = 2.5e-6

    p0 = Ekin*e/c
    gamma = 1+Ekin*e/(m_p*c**2)
    beta = np.sqrt(1-gamma**-2)
    ax = 0.92e-4
    ay = 0.96e-4
    bxy = 0.65e-4
    a = np.array(((.92e-4, -.65e-4), (-.65e-4, .96e-4)))

    @jit(nogil=True)
    def tune_dist_funcOCT(J_x, J_y):
        J = np.array((J_x, J_y))
        a = np.array(((.92e-4, -.65e-4), (-.65e-4, .96e-4)))
        I = 550
        I_max = 550
        a *= I/I_max
        return get_octupole_tune(-a, J)

    def tune_dist_funcEL(J_x, J_y):
        return get_elens_tune_for_round_beam_simplified(1e-3, J_x, J_y)

    def tune_dist_funcPEL(Jz):
        return get_pelens_tune(Jz, max_tune_shift_x=1e-3, max_tune_shift_y=1e-3)

    def tune_dist_funcRFQ(Jz):
        v2 = 2.23e9
        return get_rfq_tune(Jz, v2)
    # dispersion_solver = TransverseDispersionRelationWithSpaceCharge(tune_dist_func2, 0.2e-4)
    # dispersion_solver = TransverseDispersionRelation(tune_dist_funcOCT)
    dispersion_solver = LongitudinalDispersionRelation(tune_dist_funcRFQ)
    # dispersion_solver = GeneralizedDispersionRelation(transverse_tune_distribution_function=tune_dist_funcOCT,
    #   longitudinal_tune_distribution_function=tune_dist_funcRFQ)
    # dispersion_solver = LongitudinalDispersionRelationWithSpaceCharge(tune_dist_func_long,  0.0001)
    legend = []
    tune_vec = np.linspace(-10*Q_S, 10*Q_S, 10000)
    for mode in [0]:
        def func(Jz, mode):
            return np.power(Jz, np.abs(mode))*np.exp(-Jz)

        def normalisation(mode=0):
            # return tplquad(func, 0, MAX_INTEGRAL_LIMIT,
                        #    lambda x: 0, lambda x: MAX_INTEGRAL_LIMIT,
                        #    lambda x, y: 0, lambda x, y: MAX_INTEGRAL_LIMIT,
                        #    args=(mode,))[0]
            return quad(func, 0, MAX_INTEGRAL_LIMIT, args=(mode,))[0]
            # return 1
        N = normalisation(mode)
        print('Normalisation for mode {0:} is: {1:.2e}'.format(mode, N))
        real_vec, imag_vec = dispersion_solver.dispersion_relation(
            tune_vec, Q_S, mode=mode)
        real_vec /= N
        imag_vec /= N
    # sc_real_vec, sc_imag_vec = dispersion_solver.sc_component_dispersion_relation(tune_vec, Qs)
    # stab_vec_re, stab_vec_im = dispersion_solver.tune_shift(real_vec, imag_vec, sc_real_vec, sc_imag_vec)
        stab_vec_re, stab_vec_im = dispersion_solver.tune_shift(
            real_vec, imag_vec)
        folder = '/home/vgubaidulin/PhD/Data/DR/rfq(m={0:})/'.format(mode)
        # os.mkdir(folder)
        save_results(folder, stab_vec_re, stab_vec_im, tune_vec)
        plt.plot(stab_vec_re/Q_S, stab_vec_im/Q_S)
    plt.legend(legend, loc='upper left')
    plt.xlabel('$\Im\Delta Q$')
    plt.ylabel('$\Re\Delta Q$')

    plt.tight_layout()
    plt.show()
