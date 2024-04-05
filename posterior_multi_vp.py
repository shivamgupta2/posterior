import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import scipy
import torch

def compute_score(x, t, R):
    if len(x.shape) == 3:
        score = (-((x-R)/t) * multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5 - ((x + R)/t) * multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5)/(multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5 + multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5)
    else:
        score = (-((x-R)/t) * multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, np.newaxis] * 0.5 - ((x + R)/t) * multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, np.newaxis] * 0.5)/(multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, np.newaxis] * 0.5 + multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, np.newaxis] * 0.5)
    return score

def compute_vp_score(x, t, R):
    uncond_score = np.exp(t) * compute_score(np.exp(t) * x, np.exp(2 * t) - 1, R)
    return uncond_score


def create_time_schedule(num_steps, end_time, scale):
    schedule = np.zeros(num_steps)
    schedule[0] = end_time
    cur_time = end_time
    for i in range(1, num_steps):
        cur_time += cur_time * scale
        schedule[i] = cur_time
    schedule = schedule[::-1]
    return schedule


#vectorized log pdfs for multiple mean vectors (given in shape (num_samples, num_particles, dim)), all with same cov, with multiple x's in shape (num_samples, dim)
#could also be that x has same shape as mean vectors
#returns log pdfs in shape(num_samples, num_particles)
def vectorized_gaussian_logpdf(x, means, covariance):
    _, d = covariance.shape
    constant = d * np.log(2 * np.pi)
    _, log_det = np.linalg.slogdet(covariance)
    print('cov:', covariance, 'log det:', log_det, 'constant:', constant)
    cov_inv = np.linalg.inv(covariance)
    if x.shape == means.shape:
        deviations = x - means
    elif len(x.shape) > len(means.shape):
        deviations = x - means[:, None, :]
    else:
        print('here:', x[0], means[0,0])
        deviations = x[:, None, :] - means
        print('deviations:', deviations[0,0])
    central_term = np.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations)
    print('cov and central term:', covariance, central_term[0,0])
    res = -0.5 * central_term
    print('log prob:', res[0,0], 'prob:', np.exp(res)[0,0])
    return res

#given probs of shape(num_samples, num_particles), produce res of shape(num_samples, num_particles), where each res[i, j] \in [0, num_particles) with probability probs[i, res[i,j]]
def vectorized_random_choice(probs):
    cumulative_probs = probs.cumsum(axis=1)[:, np.newaxis, :]
    unif_samples = np.random.rand(probs.shape[0], probs.shape[1])
    res = (cumulative_probs > unif_samples[:,:,np.newaxis]).argmax(axis=2)
    return res

def twisted_diffusion(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=500, num_particles=1000):
    dim = R.shape[0]
    cond_samples = np.zeros((num_samples, dim))

    #rev_schedule = schedule[::-1]

#TODO: rewrite to take forward operator, vectorize
#edm noise schedule
def particle_filter(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=500, num_particles=1000, c=1):
    dim = R.shape[0]
    cond_samples = np.zeros((num_samples, dim))

    cur_samples = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), (num_samples,))

    rev_schedule = schedule[::-1]
    noisy_y = np.zeros((num_samples, num_steps, dim))
    noisy_y[:, -1, :] = np.dot(measurement_A, cur_samples.T).T
    for it in range(num_steps-1, 0, -1):
        end_time = rev_schedule[0]
        prev_time = rev_schedule[it]
        cur_time = rev_schedule[it-1]
        step_size = rev_schedule[it] - rev_schedule[it-1]
        bar_alpha_t = np.exp(-2 * prev_time)
        bar_alpha_t_minus_1 = np.exp(-2 * cur_time)
        p_k = np.sqrt(((1-c) * (1 - bar_alpha_t_minus_1))/(1 - bar_alpha_t))
        q_k = np.sqrt(c * (1 - bar_alpha_t_minus_1))
        noisy_y[:, it-1, :] = np.sqrt(bar_alpha_t_minus_1) * y + p_k * (noisy_y[:, it, :] - np.sqrt(bar_alpha_t) * y) + q_k * np.dot(measurement_A, np.random.multivariate_normal(np.zeros(dim), np.eye(dim), num_samples).T).T
    next_samples = np.zeros((num_samples,num_particles, dim))
    next_samples[:, np.arange(num_particles), :] = cur_samples[:, None, :]
    cur_samples = next_samples
    for it in range(num_steps-2, -1, -1):
        print('it:', it)
        step_size = rev_schedule[it + 1] - rev_schedule[it]
        prev_time = rev_schedule[it+1]
        cur_time = rev_schedule[it]

        alpha_t_plus_1 = np.exp(-2 * step_size)
        bar_alpha_t_plus_1 = np.exp(-2 * prev_time)
        bar_alpha_t = np.exp(-2 * cur_time)

        uncond_score = compute_vp_score(cur_samples, prev_time, R)

        Sigma_k_plus_1_DDIM = c * (1 - bar_alpha_t) * np.eye(dim)
        Sigma_k_plus_1_FPS = np.linalg.inv(np.linalg.inv(Sigma_k_plus_1_DDIM) + (1/(measurement_var * bar_alpha_t)) * np.dot(measurement_A.T, measurement_A))

        mu_k_plus_1_DDIM = (1/alpha_t_plus_1) * (cur_samples + (1 - bar_alpha_t_plus_1) * uncond_score) - np.sqrt((1 - c) * (1 - bar_alpha_t) * (1 - bar_alpha_t_plus_1)) * uncond_score
        
        mu_k_plus_1_FPS_helper_1 = np.einsum('ij,klj->kli', np.linalg.inv(Sigma_k_plus_1_DDIM), mu_k_plus_1_DDIM)
        mu_k_plus_1_FPS = np.einsum('ij,klj->kli', Sigma_k_plus_1_FPS, mu_k_plus_1_FPS_helper_1 + (1/(measurement_var * bar_alpha_t)) * np.dot(measurement_A.T, noisy_y[:, it, :].T).T[:, None, :])

        next_samples = np.random.multivariate_normal(np.zeros(dim), Sigma_k_plus_1_FPS, (num_samples, num_particles)) + mu_k_plus_1_FPS


        log_probs = np.zeros((num_samples, num_particles))

        #resampling particles
        print(measurement_var * bar_alpha_t)
        #log_probs_term_1 = 0
        #Ax_for_next_samples = np.einsum('ij,klj->kli', measurement_A, next_samples)
        Ax_for_next_samples = np.zeros((num_samples, num_particles, dim))
        for i in range(num_samples):
            Ax_for_next_samples[i] = np.dot(measurement_A, next_samples[i].T).T
        print('computing term 1:')
        log_probs_term_1 = vectorized_gaussian_logpdf(noisy_y[:,it,:], Ax_for_next_samples, measurement_var * bar_alpha_t * np.eye(dim))
        #log_probs_term_1 = 0
        print('computing_term 2:')
        log_probs_term_2 = vectorized_gaussian_logpdf(next_samples, mu_k_plus_1_DDIM, Sigma_k_plus_1_DDIM)
        log_probs_term_3 = vectorized_gaussian_logpdf(next_samples, mu_k_plus_1_FPS, Sigma_k_plus_1_FPS)
        log_probs = log_probs_term_1 + log_probs_term_2 - log_probs_term_3
        print('here log_probs:', log_probs.shape, log_probs[0], (log_probs - np.max(log_probs, axis=1)[:, np.newaxis])[0])

        probs = np.exp(log_probs - np.max(log_probs, axis=1)[:, np.newaxis])
        print('probs 0:', probs[0])
        #probs = np.ones(log_probs.shape)
        probs /= np.sum(probs, axis=1)[:, np.newaxis]

        resampled_indices = np.zeros((num_samples, num_particles), dtype=int)
        for i in range(num_samples):
            resampled_indices[i] = np.random.choice(num_particles, size=num_particles, p=probs[i])
            cur_samples[i] = next_samples[i][resampled_indices[i], :]

        if it % 50 == 0:
            print('it:', it, 'time:', rev_schedule[it])
            plt.scatter(cur_samples[:,0, 0], cur_samples[:,0, 1])
            plt.show()

        #cur_samples = next_samples[np.arange(num_samples)[:, np.newaxis], resampled_indices]
        #cur_samples = next_samples


    cond_samples = cur_samples[:, 0, :]
    #cond_sample_ids = np.random.choice(np.arange(num_particles), size=num_samples)
    #print('done!!')
    #cond_samples = cur_samples[np.arange(num_samples), cond_sample_ids]
    #print(cond_samples[1], cur_samples[1])
    return cond_samples

def annealed_uncond_langevin(R, schedule, num_steps, num_samples=50, c=1):
    start_time = schedule[0]
    end_time = schedule[len(schedule)-1]
    uncond_samples = np.random.multivariate_normal(np.zeros(2), np.eye(2), num_samples)
    for it in range(1, num_steps):
        cur_time = schedule[it]
        prev_time = schedule[it-1]
        step_size = schedule[it-1] - schedule[it]
        uncond_score = compute_vp_score(uncond_samples, prev_time, R)
        beta_t = 1 - np.exp(-2 * step_size)
        alpha_t = np.exp(-2 * step_size)
        bar_alpha_t = np.exp(-2 * prev_time)
        bar_alpha_t_minus_1 = np.exp(-2 * cur_time)
        uncond_samples = (1/alpha_t) * (uncond_samples + (1 - bar_alpha_t) * uncond_score) - np.sqrt((1 - c) * (1 - bar_alpha_t_minus_1) * (1 - bar_alpha_t)) * uncond_score\
                + np.random.multivariate_normal(np.zeros(2), c * (1 - bar_alpha_t_minus_1) * np.eye(2), num_samples)
        #uncond_samples = (1/np.sqrt(alpha_t)) * uncond_samples + (beta_t/np.sqrt(alpha_t)) * uncond_score + np.random.multivariate_normal(np.zeros(2), beta_t * (1 - bar_alpha_t_minus_1)/(1 - bar_alpha_t) * np.eye(2), num_samples)
        #uncond_samples =  (2 - np.sqrt(1 -  beta_t)) * uncond_samples + beta_t * uncond_score + np.random.multivariate_normal(np.zeros(2), beta_t * np.eye(2), num_samples)
        if it % 50 == 0:
            print('it:', it, 'time:', schedule[it])
            #plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1])
            #plt.show()
    return uncond_samples

def rejection_sampler(R, schedule, num_steps, num_samples, y, meas_A, meas_var):
    done = np.zeros(num_samples, dtype=int)
    cond_samples = np.zeros((num_samples, R.shape[0]))
    while np.sum(done) < num_samples:
        uncond_samples = annealed_uncond_langevin(R, schedule, num_steps, num_samples)
        Ax = np.dot(meas_A, uncond_samples.T).T
        accept_prob = np.exp(-np.linalg.norm(Ax - y, axis=1)**2/(2 * meas_var))
        unif_samples = np.random.rand(num_samples)
        cond_samples[unif_samples < accept_prob] = uncond_samples[unif_samples < accept_prob]
        done[unif_samples < accept_prob] = np.ones(num_samples, dtype=int)[unif_samples < accept_prob]
        print(cond_samples[done == 1].shape)
    return cond_samples


def vectorized_gaussian_logpdf_single_mean(x, mean, covariance):
    _, d = covariance.shape
    constant = d * np.log(2 * np.pi)
    _, log_det = np.linalg.slogdet(covariance)
    cov_inv = np.linalg.inv(covariance)
    deviations = x - mean
    print(deviations.shape)
    central_term = np.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations)
    print('here:', central_term.shape)
    return -0.5 * central_term

#Parameters
R = np.ones(2)
num_steps = 500
end_time = 0.01
num_particles=5
num_samples = 500
schedule = create_time_schedule(num_steps, end_time, 0.02)
print(schedule)


#print('done with uncond')

meas_A = np.array([[0, 0], [0, 1]])
meas_var = 1
#meas_var = 1e-2
meas_y = np.array([0, 0.8])
#plt.savefig(str(num_particles) + '_particles.pdf')
#plt.show()

bar_alpha_end_time = np.exp(-2 * end_time)
uncond_samples = annealed_uncond_langevin(R, schedule, num_steps, num_samples=num_samples)
#rej_samples = rejection_sampler(R, schedule, num_steps, num_samples, meas_y, meas_A, meas_var)
plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1], label='uncond_samples')
#plt.scatter(rej_samples[:, 0], rej_samples[:, 1], label='Rejection Sampling')
plt.legend()
#true_samples = np.random.multivariate_normal(R, end_time * np.eye(2), 3000)
#plt.scatter(true_samples[:, 0], true_samples[:, 1])
plt.show()

x, y = np.mgrid[-20:20:0.01, -20:20:0.01]
pos = np.dstack((x, y))
print('pos:', pos.shape)
#uncond_density1 = 0.5 * multivariate_normal.pdf(pos, R, end_time * np.eye(2)) + 0.5 * multivariate_normal.pdf(pos, -R, end_time * np.eye(2))
#uncond_density2 = np.log(0.5) + scipy.special.logsumexp((multivariate_normal.logpdf(pos, R, end_time * np.eye(2)), multivariate_normal.logpdf(pos, -R, end_time * np.eye(2))))
uncond_density = np.log(0.5) + np.logaddexp(vectorized_gaussian_logpdf_single_mean(pos, R, end_time * np.eye(2)), vectorized_gaussian_logpdf_single_mean(pos, -R, end_time * np.eye(2)))
#uncond_density = np.exp(uncond_density)
#uncond_density /= np.sum(uncond_density)
#print('uncond density shape:', uncond_density.shape)
#print('densisites:', np.sum(np.exp(uncond_density)))
cond_samples = particle_filter(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)

Ax = np.copy(pos)
Ax[:,:,0] = 0
#p_y_cond_x = multivariate_normal.logpdf(Ax, meas_y, meas_var * np.eye(2))
p_y_cond_x = vectorized_gaussian_logpdf_single_mean(Ax, meas_y, bar_alpha_end_time * meas_var * np.eye(2))
#print('p_y_cond_x_shape:', p_y_cond_x.shape)
#p_y = 0.5 * norm.pdf(meas_y[1], R[1], np.sqrt(end_time + meas_var)) + 0.5 * norm.pdf(meas_y[1], -R[1], np.sqrt(end_time + meas_var))
#print('here:', p_y)

#cond_density = uncond_density * p_y_cond_x
cond_density = np.exp(uncond_density + p_y_cond_x)
cond_density /= np.sum(cond_density)
flat_density = cond_density.flatten()
#flat_uncond_density = uncond_density.flatten()
sample_index = np.random.choice(np.arange(len(x) * len(y)), p=flat_density, size=num_samples, replace=False)

selections = pos.reshape(-1, 2)[sample_index]

plt.scatter(selections[:, 0], selections[:, 1], label='True Distribution')

plt.scatter(cond_samples[:, 0], cond_samples[:, 1], label='Particle Filter')
plt.title('num particles = ' + str(num_particles))

rej_samples = rejection_sampler(R, schedule, num_steps, num_samples, meas_y, meas_A, meas_var * bar_alpha_end_time)
plt.scatter(rej_samples[:, 0], rej_samples[:, 1], label='Rejection Sampling')

plt.legend()
plt.savefig(str(num_particles) + '_particles_vectorized.pdf')
plt.show()
