import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

with open('result-data/1d-hist-1-1000000-50.pickle', 'rb') as f:
    data = pickle.load(f)


rej_samples = data['rej_samples']
x_hist = data['x_hist']
ind_hist = data['ind_hist']
schedule = data['schedule']

num_samples, num_particles, num_steps = x_hist.shape

cond_samples = x_hist[:,:,-1]


if num_particles == 10**5:
    num_particles_name = '10^5'
elif num_particles == 10**6:
    num_particles_name = '10^6'
else:
    num_particles_name = f'{num_particles}'

plt.ion()

bins = np.arange(-1.0, 1.61, 0.05)  #20
plt.hist(cond_samples.flatten(), bins=bins, density=True, histtype='step', label=f'{num_particles_name} particles')
plt.hist(rej_samples, bins=bins, density=True, histtype='step', label='Rejection Sampling')
plt.xlabel('$x$')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.title('Posterior Sampling Distribution')
#plt.show()
plt.savefig('1d-samples.pdf')


i=55
plt.clf()
locs= ind_hist[0,:,i][np.where(x_hist[0,:,-1] < -.5)];xxall = x_hist[0,ind_hist[0,:,i],i];xx = x_hist[0,locs,i];xxthen = x_hist[0,:,i]
bins = np.arange(-2, 3.01, 0.1)
plt.hist(xxthen, bins=bins, density=True, histtype='step', label='Particles at $t=0.4$')
plt.hist(xxall, bins=bins, density=True, histtype='step', label='$t=0.4$ preimage of final particles')
plt.title('Bottleneck at $t=.4$')
plt.legend(loc='upper left')
plt.ylim([0, 1.3])
plt.xlabel('$x$')
plt.ylabel('Density')

t=-1.;
print(f'Threshold {t}: {np.sum(xxthen < t)} particles turned into {np.sum(xxall < t)}, for {np.mean(xxall < t) / np.mean(xxthen < t):.2f}x growth')
print(f'    (from {np.mean(xxthen < t)*100:.2f}% to {np.mean(xxall < t)*100:.2f}%)')
plt.savefig('1d-preimage.pdf')

if '-i' in sys.argv:
    import code
    d = globals()
    d.update(locals())
    code.interact(local=d)

