from tune_calculation import *
if __name__=='__main__':
    Ekin = 8.602e6
    n_particles = 8*2048
    dQrms_elens_x = np.empty(shape=(5,), dtype=np.float64)
    Jz = trunc_exp_rv(0, 2.5**2
    , 1.0, n_particles)
    p0 = Ekin*e/c
    gamma = 1+Ekin*e/(m_p*c**2)
    beta = np.sqrt(1-gamma**-2)
    epsx = 0.25*35e-6
    epsy = 0.25*15e-6