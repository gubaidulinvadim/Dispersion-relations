from seaborn import palettes
from tune_calculation import *
from tqdm import tqdm
if __name__ == '__main__':
    n_particles = int(1e5)
    Jx = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    Jy = trunc_exp_rv(low=0, high=4, scale=1.0, size=n_particles)
    points = 16
    rms_x = np.empty(shape=(points,), dtype=np.float64)
    rms_y = np.empty(shape=(points,), dtype=np.float64)
    mean_x = np.empty(shape=(points,), dtype=np.float64)
    mean_y = np.empty(shape=(points,), dtype=np.float64)
    dQmax = 1e-3
    ratios = np.linspace(0.5, 2., points)
    for index, ratio in tqdm(enumerate(ratios)):
        dQxel, dQyel = get_elens_tune(
            dQmax=dQmax, Jx=Jx, Jy=Jy, ratio=ratio, simplified=False)
        mean_x[index], mean_y[index] = np.mean(dQxel), np.mean(dQyel)
        rms_x[index], rms_y[index] = np.sqrt(
            np.var(dQxel)), np.sqrt(np.var(dQyel))
    palette = sbs.color_palette('RdBu')
    fig, ax = plt.subplots(1, 1)
    ax.plot(ratios, mean_x/dQmax, label='Coherent tune shift')
    ax.plot(ratios, rms_x/dQmax, label='RMS tune spread')
    plt.figlegend(loc='lower right', frameon=False)
    plt.savefig(
        'Results/'+'GS-elens-sizes'+'.pdf', bbox_inches='tight')
    plt.show()
