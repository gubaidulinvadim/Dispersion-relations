from Schenk import *
from scipy.constants import c, m_p, e
from parameters.SIS100_constants import *
if __name__ == '__main__':
    sbs.set(rc={'figure.figsize': (8.3, 5.2),
                'text.usetex': True,
                'font.family': 'serif',
                'font.size': 20,
                'axes.linewidth': 2,
                'lines.linewidth': 3,
                'legend.fontsize': 16,
                'legend.numpoints': 1, },
            style='ticks',
            palette='colorblind',
            context='talk')
    Jz = np.linspace(0, 3, 20)

    print('beta_z: ', BETA_Z)
    Jz *= SIGMA_Z**2/(2*BETA_Z)
    print('Order of magnitude for longitudinal amplitude: ',
          SIGMA_Z/(BETA*c)*OMEGA_REV)
    sbs.set_palette('colorblind')
    time_start = time.process_time()
    p = 0
    l = 1
    Hr, Hi = H(Jz, p=p, l=l)
    time_elapsed = time.process_time()-time_start
    print('Time elapsed: {0:.2e}'.format(time_elapsed))
    plt.plot(Jz/(SIGMA_Z**2/(2*BETA_Z)), np.sqrt(Hr**2+Hi**2), c='b',
             linewidth=2, marker='o', markersize=2, label='$|H^0_l(J_z)|$')
    omega_0 = c/CIRCUMFERENCE
    Q_X = 18.86
    omega_p = OMEGA_REV*(p+Q_X+l*Q_S)
    plt.plot(Jz/(SIGMA_Z**2/(2*BETA_Z)),
             np.abs(jv(l, np.sqrt(2*Jz*BETA_Z)/c*(omega_p))), c='r', linewidth=1, marker='o', markersize=1, label='$|J_l(J_z)|$')
    plt.xlabel('$J_z/\epsilon_z$')
    plt.ylabel('Specrtal function')
    plt.legend(frameon=False)
    plt.savefig('Results/'+'SIS100H_{0:}.pdf'.format(l), bbox_inches='tight')
    plt.show()
