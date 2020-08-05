import numpy as np
import scipy as sp
from scipy.constants import epsilon_0, c, m_e, m_p, e, pi
from scipy.special import i0, i1
from scipy.integrate import quad, dblquad
from joblib import Parallel, delayed
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sbs


from scipy.special import iv
from tune_calculation import *


class TransverseDispersionRelation():
    def __init__(self, tune_distribution_function, epsilon=1e-6, max_integral_limit=16):
        self.function = tune_distribution_function
        self.epsilon = epsilon
        self.max_int = max_integral_limit

    def distribution_func(self, J_x, J_y):
        return -np.exp(-J_x-J_y)

    def dispersion_integrand(self, J_x, J_y, tune: float, Qs, mode=0):
        '''
        J_x: horizontal action variable normalized to emittance
        J_y: vertical action variable normalized to emittance
        tune: dimensionless tune corresponding to frequency of external excitation
        '''
        tune_shift_x, tune_shift_y = self.function(J_x, J_y)
        return -self.distribution_func(J_x, J_y)*J_x/(tune - tune_shift_x - mode*Qs + 1j*self.epsilon)

    def real_part_of_integrand(self, *args):
        return np.real(self.dispersion_integrand(*args))

    def imaginary_part_of_integrand(self, *args):
        return np.imag(self.dispersion_integrand(*args))

    def compute_real_part(self, tune, Qs, mode=0):
        r = dblquad(self.real_part_of_integrand, 0., self.max_int,
                    lambda x: 0., lambda x: self.max_int, args=(tune, Qs, mode))[0]
        return r

    def compute_imag_part(self, tune, Qs, mode=0):
        i = dblquad(self.imaginary_part_of_integrand, 0., self.max_int,
                    lambda x: 0., lambda x: self.max_int, args=(tune, Qs, mode))[0]
        return i

    def dispersion_relation(self, tune_vec, Qs, mode=0):
        real_vec = np.empty(shape=(len(tune_vec),), dtype=np.float64)
        imag_vec = np.empty(shape=(len(tune_vec), ), dtype=np.float64)
        for (j, tune) in tqdm(enumerate(tune_vec), total=len(tune_vec)):
            real_vec[j] = self.compute_real_part(tune, Qs, mode)
            imag_vec[j] = self.compute_imag_part(tune, Qs, mode)
        return real_vec, imag_vec

    def get_tune_shift(self, real_vec, imag_vec):
        ampl_vec = real_vec*real_vec+imag_vec*imag_vec
        stab_vec_re = np.divide(real_vec, ampl_vec)
        stab_vec_im = np.divide(-imag_vec, ampl_vec)
        return stab_vec_re, stab_vec_im


class LongitudinalDispersionRelation(TransverseDispersionRelation):
    def __init__(self, tune_distribution_function, epsilon=1e-6, max_integral_limit=16):
        super().__init__(tune_distribution_function, epsilon, max_integral_limit)

    def distribution_func(self, Jz):
        return np.exp(-Jz)

    def dispersion_integrand(self, Jz, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jz)
        return self.distribution_func(Jz)*Jz**(mode)/(tune-tune_x-mode*Qs+1j*self.epsilon)

    def compute_real_part(self, tune, Qs, mode=0):
        r = quad(self.real_part_of_integrand, 0.,
                 self.max_int, args=(tune, Qs, mode))[0]
        return r

    def compute_imag_part(self, tune, Qs, mode=0):
        i = quad(self.imaginary_part_of_integrand, 0.,
                 self.max_int, args=(tune, Qs, mode))[0]
        return i
# class TransverseDispersionRelationWithSpaceCharge(TransverseDispersionRelation):
#     def __init__(tune_distribution_function, epsilon=1e-6, max_integral_limit=16):
#         return super().__init__(tune_distribution_function, epsilon, max_integral_limit)


class LongitudinalDispersionRelationWithSpaceCharge(LongitudinalDispersionRelation):
    def __init__(self, tune_distribution_function, dQmax_sc, epsilon=1e-6, max_integral_limit=16):
        super().__init__(tune_distribution_function, epsilon, max_integral_limit)
        self.dQmax_sc = dQmax_sc
        self.sc_function = get_pelens_tune

    def dispersion_integrand(self, Jz, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jz)
        sc_tune_x, sc_tune_y = self.sc_function(Jz, -self.dQmax_sc)
        return self.distribution_func(Jz)*Jz**(mode)/(tune-tune_x-sc_tune_x-mode*Qs+1j*self.epsilon)

    def sc_dispersion_integrand(self, Jz, tune, Qs, mode=0):
        tune_x, tune_y = self.function(Jz)
        sc_tune_x, sc_tune_y = self.sc_function(Jz, -self.dQmax_sc)
        return sc_tune_x*self.distribution_func(Jz)*Jz**(mode)/(tune-tune_x-sc_tune_x-mode*Qs+1j*self.epsilon)

    def sc_real_part_of_integrand(self, *args):
        return np.real(self.sc_dispersion_integrand(*args))

    def sc_imag_part_of_integrand(self, *args):
        return np.imag(self.sc_dispersion_integrand(*args))

    def sc_compute_real_part(self, tune, Qs):
        r = quad(self.sc_real_part_of_integrand, 0,
                 self.max_int, args=(tune, Qs))[0]
        return r

    def sc_compute_imag_part(self, tune, Qs):
        i = quad(self.sc_imag_part_of_integrand, 0,
                 self.max_int, args=(tune, Qs))[0]
        return i

    def sc_component_dispersion_relation(self, tune, Qs):
        sc_real_vec = np.empty(shape=(len(tune_vec),), dtype=np.float64)
        sc_imag_vec = np.empty(shape=(len(tune_vec), ), dtype=np.float64)
        for (j, tune) in tqdm(enumerate(tune_vec), total=len(tune_vec)):
            sc_real_vec[j] = self.sc_compute_real_part(tune, Qs)
            sc_imag_vec[j] = self.sc_compute_imag_part(tune, Qs)
        return sc_real_vec, sc_imag_vec

    def get_tune_shift(self, real_vec, imag_vec, sc_real_vec, sc_imag_vec):
        ampl_vec = real_vec*real_vec+imag_vec*imag_vec
        # stabilityx_vec=divide(real_vec+real_vec*real_sc_vec+imag_vec*imag_sc_vec,absint_vec)
        # stabilityy_vec=divide(-imag_vec-imag_vec*real_sc_vec+real_vec*imag_sc_vec,absint_vec)
        stab_vec_re = np.divide(
            real_vec+real_vec*sc_real_vec+imag_vec*sc_imag_vec, ampl_vec)
        stab_vec_im = np.divide(-imag_vec-imag_vec *
                                sc_real_vec+real_vec*sc_imag_vec, ampl_vec)
        return stab_vec_re, stab_vec_im


def save_results(folder, stab_vec_re, stab_vec_im, tune_vec):
    np.save(
        folder+'dQre_x.npy', stab_vec_re)
    np.save(
        folder+'dQim_X.npy', stab_vec_im)
    np.save(
        folder+'tunes_x.npy', tune_vec)


if __name__ == '__main__':
    @jit(nogil=True)
    def tune_dist_func(J_x, J_y):
        Ekin = 7e12
        epsn = 2.2e-6

        p0 = Ekin*e/c
        gamma = 1+Ekin*e/(m_p*c**2)
        # a1 = epsn/gamma * \
        # get_octupole_coefficients(p0, 140/72*175.5, 140/72*33.6, 63100, 25*84, 0.32)
        # a2 = epsn/gamma * \
        # get_octupole_coefficients(p0, 140/72*30.1, 140/72*178.8, 63100, 25*84, 0.32)
        a1 = epsn/gamma * \
            get_octupole_coefficients(p0, 175.5, 33.6, 63100, 84, 0.32)
        a2 = epsn/gamma * \
            get_octupole_coefficients(p0, 30.1, 178.8, 63100, 84, 0.32)
        a = a1 + a2
        J = np.array((J_x, J_y))
        return get_octupole_tune(a, J)

    def tune_dist_func2(J_x, J_y):
        return get_elens_tune(0.002, J_x, J_y)

    def tune_dist_func3(J_x, J_y):
        return tune_dist_func(J_x, J_y)+tune_dist_func2(J_x, J_y)

    def pulsed_elens_tune(J_z, max_tune_shift, sigma_ez, beta_z):
        # K = J_z*beta_z/(2*sigma_ez**2)
        # dQx, err = scipy.integrate.quad(pulsed_integrand_func, 0, 1, args=(K,))
        # dQy, err = scipy.integrate.quad(pulsed_integrand_func, 0, 1, args=(K,))
        dQx = np.exp(-Jz)*iv(0, Jz)
        dQy = dQx
        return max_tune_shift*dQx, max_tune_shift*dQy

    def tune_dist_func_long(Jz):
        # beta_z = 670
        # sigma_z = 0.06
        max_tune_shift = 0.0015
        return get_pelens_tune(Jz, max_tune_shift)
    # dispersion_solver = TransverseDispersionRelation(tune_dist_func)
    # dispersion_solver = LongitudinalDispersionRelation(tune_dist_func_long)
    Qs = 0.001
    modes = np.linspace(0, 3, 4)
    legend = []
    # for mode in modes:
    #     tune_vec = np.linspace(-2.e-3+mode*Qs, 2.e-3+mode*Qs, 5000)
    #     real_vec, imag_vec = dispersion_solver.dispersion_relation(tune_vec, Qs, mode=mode)
    #     stab_vec_re, stab_vec_im = dispersion_solver.get_tune_shift(
    #         real_vec, imag_vec)
    #     plt.plot(stab_vec_re/1e-3, stab_vec_im/1e-3, marker=None)
    #     legend.append('Mode {}'.format(int(mode)))
    ratios = np.linspace(0., 0.9, 1)
    for ratio in ratios:
        tune_vec = np.linspace(-5e-3, 5e-3, 5000)
        dispersion_solver = LongitudinalDispersionRelationWithSpaceCharge(
            tune_dist_func_long, ratio*1e-3)
        real_vec, imag_vec = dispersion_solver.dispersion_relation(
            tune_vec, Qs)
        sc_real_vec, sc_imag_vec = dispersion_solver.sc_component_dispersion_relation(
            tune_vec, Qs)
        stab_vec_re, stab_vec_im = dispersion_solver.get_tune_shift(
            real_vec, imag_vec, sc_real_vec, sc_imag_vec)
        legend = ('Compensation ratio: {0:.2f}'.format(ratio),)
        plt.plot(stab_vec_re/1e-3, stab_vec_im/1e-3, marker=None)
    np.save('/home/vgubaidulin/PhD/Data/Stability_diagrams/pelens/dQre.npy', stab_vec_re)
    np.save('/home/vgubaidulin/PhD/Data/Stability_diagrams/pelens/dQim.npy', stab_vec_im)
    np.save('/home/vgubaidulin/PhD/Data/Stability_diagrams/pelens/tunes.npy', tune_vec)
    plt.xlim(-1, 1)
    plt.ylim(0, 0.25)
    plt.legend(legend)
    plt.show()
