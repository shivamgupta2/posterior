import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import sys
import glob
from scipy import stats

with open('result-data/1d-hist-1-1000000-50.pickle', 'rb') as f:
    data = pickle.load(f)
    rej_samples = data['rej_samples']
    x_hist = data['x_hist']
    ind_hist = data['ind_hist']
    schedule = data['schedule']
    num_samples, num_particles, num_steps = x_hist.shape
    cond_samples = x_hist[:,:,-1]


def load_cond(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        return data['cond_samples']
    


if num_particles == 10**5:
    num_particles_name = '10^5'
elif num_particles == 10**6:
    num_particles_name = '10^6'
else:
    num_particles_name = f'{num_particles}'

plt.ion()
plt.figure(figsize=(6, 3))


def hist(samples, bins=None, **kws):
    xlim = [np.min(samples), np.max(samples)]
    radius = None
    if bins is not None:
        xlim = [np.min(bins), np.max(bins)]
        #radius = bins[1] - bins[0]
    x = np.linspace(xlim[0], xlim[1], 1000)
    kde = stats.gaussian_kde(samples, radius)
    plt.plot(x, kde(x), label=kws.get('label'))


toplot = []
#toplot.append('TDS-samples')
#toplot.append('TDS-prior')
#toplot.append('Song-samples')
toplot.append('TDS-prior')


if 'TDS-samples' in toplot:
    samples100 = load_cond('result-data/1d-nohist-10000-100-50.pickle')
    samples1 = load_cond('result-data/1d-nohist-1000000-1-50.pickle')

    bins = np.arange(-1.0, 1.61, 0.05)  #20
    hist(rej_samples, bins=bins, density=True, histtype='step', label='Rejection Sampling')
    hist(samples1.flatten(), bins=bins, density=True, histtype='step', label=f'DPS (1 particle)')
    hist(samples100.flatten(), bins=bins, density=True, histtype='step', label=f'100 particles')
    hist(cond_samples.flatten(), bins=bins, density=True, histtype='step', label=f'$10^6$ particles')
    plt.xlabel('$x$')
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    plt.ylim([0, None])
    #plt.yticks([])
    plt.title('Sample Distribution')
    plt.tight_layout()
    #plt.show()
    plt.savefig('1d-samples.pdf')

if 'TDS-prior' in toplot:
    i=55
    plt.clf()
    locs= ind_hist[0,:,i][np.where(x_hist[0,:,-1] < 0)];xxall = x_hist[0,ind_hist[0,:,i],i];xx = x_hist[0,locs,i];xxthen = x_hist[0,:,i]
    
    bins = np.arange(-2, 3.01, 0.1)
    hist(xxthen, bins=bins, density=True, histtype='step', label='$10^6$ Particles at $t=0.4$')
    hist(xxall, bins=bins, density=True, histtype='step', label='$t=0.4$ preimage of final particles')
    #hist(xx, bins=bins, density=True, histtype='step', label='$t=0.4$ preimage of left mode')
    plt.title('Bottleneck at $t=.4$')
    plt.legend(loc='upper left')
    plt.ylim([0, 1.3])
    plt.xlabel('$x$')
    plt.ylabel('Density')
    t=-1.;
    print(f'Threshold {t}: {np.sum(xxthen < t)} particles turned into {np.sum(xxall < t)}, for {np.mean(xxall < t) / np.mean(xxthen < t):.2f}x growth')
    print(f'    (from {np.mean(xxthen < t)*100:.2f}% to {np.mean(xxall < t)*100:.2f}%)')
    plt.tight_layout()
    plt.savefig('1d-preimage.pdf')


if 'Song-samples' in toplot:
    songright = load_cond('result-data/1d-nohist-songright-2-1000000-50.pickle')
    songwrong = load_cond('result-data/1d-nohist-songwrong-2-1000000-50.pickle')

    songdata = {}
    for i in (1, 10, 100, 10**3, 10**4, 10**5):
        fname = glob.glob(f'result-data/1d-nohist-songright-*-{i}-50.pickle')[0]
        songdata[i] = load_cond(fname)
        print(f'{i}: {np.mean(songdata[i] < 0)}')
    
    bins = np.arange(-1.0, 1.61, 0.05)  #20
    hist(rej_samples, bins=bins, density=True, histtype='step', label='True Posterior')
    #hist(cond_samples.flatten(), bins=bins, density=True, histtype='step', label=f'TDS')
    hist(songwrong.flatten(), bins=bins, density=True, histtype='step', label=f'FPS-SMC (as published)')
    hist(songright.flatten(), bins=bins, density=True, histtype='step', label=f'FPS-SMC (corrected)')
    #hist(cond_samples.flatten(), bins=bins, density=True, histtype='step', label=f'$10^6$ particles')
    plt.xlabel('$x$')
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    plt.ylim([0, None])
    #plt.yticks([])
    plt.title('Sampling with FPS-SMC, $10^6$ particles')
    plt.tight_layout()
    #plt.show()
    plt.savefig('1d-song.pdf')
    print(f'   Wrong: {np.mean(songwrong < 0)*100:.3f}% Right: {np.mean(songright < 0)*100:.3f}%')
    

def load_count(method, particles):
    fnames1 = glob.glob(f'result-data/1d-nohist-{method}-*-{particles}-50.pickle')
    fnames2 = glob.glob(f'result-data/1d-means-{method}-{particles}-100.txt')
    fnames2 += glob.glob(f'result-data/1d-means-{method}-{particles}-50.txt')
    left, tot = 0, 0
    if fnames1:
        data = load_cond(fnames1[0])
        left += np.sum(data < 0)
        tot += data.size
    if fnames2:
        print(fnames2)
        with open(fnames2[0]) as f:
            for line in f:
                a, b = map(int, line.split())
                left += a
                tot += b
    return left, tot
        

    
if 'Both-means' in toplot:
    truth = 0.11802437851168306
    print(f'Truth: {truth*100:.3f}% rej: {np.mean(rej_samples < 0)*100:.3f}')

    songdata = {}
    for i in (1, 10, 100, 10**3, 10**4, 10**5, 10**6, 10**7):
        songdata[i] = load_count('songright', i)
        print(f'{i}: {songdata[i][0] / songdata[i][1]}')

    tdsdata = {}
    for i in (1, 10, 100, 10**3, 10**4, 10**5, 10**6, 10**7):
        tdsdata[i] = load_count('tds', i)
        print(f'{i}: {tdsdata[i][0] / tdsdata[i][1]}')

    def getkeyvals(d):
        keys = np.array(sorted(d.keys()))
        #vals = np.array([np.mean(d[k] < 0) for k in keys])
        samples = [d[k] for k in keys]
        vals = np.array([l/t for l,t in samples])
        ints = 2*np.array([l**.5/t for l,t in samples])
        print(ints)
        return keys, vals, ints*1

    plt.errorbar(*getkeyvals(tdsdata), label='TDS', marker='x')
    plt.errorbar(*getkeyvals(songdata), label='FPS-SMC (corrected)', marker='o')
    plt.hlines(truth, -10**6, 10**10, color='red', label='True posterior')
    plt.xlim([1, 10**7])
    plt.xscale('log')
    plt.ylim([0, None])
    plt.xlabel('Number of particles')
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    #plt.yticks([])
    plt.title('Probability of sampling left mode')
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.show()
    plt.savefig('1d-mode.pdf')

                    


if '-i' in sys.argv:
    import code
    d = globals()
    d.update(locals())
    code.interact(local=d)

