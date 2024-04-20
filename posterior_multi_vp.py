import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import logsumexp
import numpy.linalg as la
import scipy
import torch

p = np.float64(0.5)
d = 2
R = 10

mu1 = np.zeros((1,d), dtype=np.float64)
mu2 = np.array([[10., 1.]], dtype=np.float64)

var_x = np.array([[.1**2, .1**2]],dtype=np.float64)
var_y = np.float64(1e-10)

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
    #print('cov:', covariance, 'log det:', log_det, 'constant:', constant)
    cov_inv = np.linalg.inv(covariance)
    if x.shape == means.shape:
        deviations = x - means
    elif len(x.shape) > len(means.shape):
        deviations = x - means[:, None, :]
    elif len(x.shape) + 2 == len(means.shape):
        #print('here:', x[0], means[0,0])
        deviations = x[None, None, :] - means
        #print('deviations:', deviations[0,0])
    else:
        deviations = x[:,None, :] - means
    central_term = np.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations)
    #print('cov and central term:', covariance, central_term[0,0])
    res = -0.5 * central_term
    #print('log prob:', res[0,0], 'prob:', np.exp(res)[0,0])
    return res

def vectorized_gaussian_score(x, means, var):
    if x.shape == means.shape:
        deviations = -(x - means)
    else:
        deviations = -(x[np.newaxis,np.newaxis,:] - means)
    return deviations/var
#given probs of shape(num_samples, num_particles), produce res of shape(num_samples, num_particles), where each res[i, j] \in [0, num_particles) with probability probs[i, res[i,j]]
def vectorized_random_choice(probs):
    cumulative_probs = probs.cumsum(axis=1)[:, np.newaxis, :]
    unif_samples = np.random.rand(probs.shape[0], probs.shape[1])
    res = (cumulative_probs > unif_samples[:,:,np.newaxis]).argmax(axis=2)
    return res

def twisted_diffusion(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=500, num_particles=1000):
    dim = R.shape[0]
    cond_samples = np.random.normal(0, np.sqrt(schedule[0]), size=(num_samples, num_particles, dim))
    cur_time = schedule[0]
    x_0_given_x_t = cond_samples + cur_time * compute_score(cond_samples, cur_time, R)
    log_Tilde_p_T = vectorized_gaussian_logpdf(y, np.einsum('ij,klj->kli', measurement_A, x_0_given_x_t), measurement_var * np.eye(dim))
    Tilde_p_T_scores = vectorized_gaussian_score(y, np.einsum('ij,klj->kli', measurement_A, x_0_given_x_t), measurement_var)

    log_w = log_Tilde_p_T
    
    for it in range(1, len(schedule)):
        cur_time = schedule[it]
        #print(cur_time)
        step_size = schedule[it-1] - schedule[it]
        log_w -= logsumexp(log_w, axis=1)[:, np.newaxis]
        w = np.exp(log_w)
        #print('w:', np.sum(w[0]))
        resampled_indices = np.zeros((num_samples, num_particles), dtype=int)
        for i in range(num_samples):
            resampled_indices[i] = np.random.choice(num_particles, size=num_particles, p=w[i])
            cond_samples[i] = cond_samples[i][resampled_indices[i], :]
            log_Tilde_p_T[i] = log_Tilde_p_T[i][resampled_indices[i]]
            Tilde_p_T_scores[i] = Tilde_p_T_scores[i][resampled_indices[i], :]
        
        uncond_score = compute_score(cond_samples, cur_time, R)
        x_0_given_x_t = cond_samples + cur_time * uncond_score
        cond_score_approx = uncond_score + Tilde_p_T_scores

        next_cond_samples = cond_samples + step_size * cond_score_approx + np.random.multivariate_normal(np.zeros(2), step_size * np.eye(2), (num_samples, num_particles))
        next_log_Tilde_p_T = vectorized_gaussian_logpdf(y, np.einsum('ij,klj->kli', measurement_A, x_0_given_x_t), measurement_var * np.eye(dim))

        log_w_term_1 = vectorized_gaussian_logpdf(next_cond_samples, cond_samples + step_size * uncond_score, step_size * np.eye(dim))

        #We have Pr[x_t | x_{t+1}, y] = N(x_t; x_{t+1} + \sigma^2 \grad \log p(x_{t+1} | y), sigma^2)
        #the score \grad \log p(x_{t+1} | y) is approximated as our cond_score_approx given above
        log_w_term_3 = vectorized_gaussian_logpdf(next_cond_samples, cond_samples + step_size * cond_score_approx, step_size * np.eye(dim))
        log_w = log_w_term_1 + next_log_Tilde_p_T - log_w_term_3 - log_Tilde_p_T

        cond_samples = next_cond_samples
        log_Tilde_p_T = next_log_Tilde_p_T

    return cond_samples[:,0,:] 

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
        #print(measurement_var * bar_alpha_t)
        Ax_for_next_samples = np.zeros((num_samples, num_particles, dim))
        for i in range(num_samples):
            Ax_for_next_samples[i] = np.dot(measurement_A, next_samples[i].T).T
        log_probs_term_1 = vectorized_gaussian_logpdf(noisy_y[:,it,:], Ax_for_next_samples, measurement_var * bar_alpha_t * np.eye(dim))
        log_probs_term_2 = vectorized_gaussian_logpdf(next_samples, mu_k_plus_1_DDIM, Sigma_k_plus_1_DDIM)
        log_probs_term_3 = vectorized_gaussian_logpdf(next_samples, mu_k_plus_1_FPS, Sigma_k_plus_1_FPS)
        log_probs = log_probs_term_1 + log_probs_term_2 - log_probs_term_3

        probs = np.exp(log_probs - np.max(log_probs, axis=1)[:, np.newaxis])
        probs /= np.sum(probs, axis=1)[:, np.newaxis]

        resampled_indices = np.zeros((num_samples, num_particles), dtype=int)
        for i in range(num_samples):
            resampled_indices[i] = np.random.choice(num_particles, size=num_particles, p=probs[i])
            cur_samples[i] = next_samples[i][resampled_indices[i], :]


    cond_samples = cur_samples[:, 0, :]
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
        #print(cond_samples[done == 1].shape)
    return cond_samples


def vectorized_gaussian_logpdf_single_mean(x, mean, covariance):
    _, d = covariance.shape
    constant = d * np.log(2 * np.pi)
    _, log_det = np.linalg.slogdet(covariance)
    cov_inv = np.linalg.inv(covariance)
    deviations = x - mean
    #print(deviations.shape)
    central_term = np.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations)
    #print('here:', central_term.shape)
    return -0.5 * central_term



#fps = FPS()
#M_fps, B_fps = 20, 1000
#y = np.array([[0.7]])
#result = fps.sample(y, M_fps, B_fps)
#randint = np.random.randint(M_fps, size=B_fps)
#x_particle = np.zeros((B_fps,d))
#for i, idx in enumerate(randint):
#    x_particle[i-1] = result[i-1, idx]
#plt.scatter(x_particle[:,0], x_particle[:,1])
#plt.show()

#Parameters
R = np.ones(2)
num_steps = 500
end_time = 0.01
num_particles=100
num_samples = 500
schedule = create_time_schedule(num_steps, end_time, 0.02)
#print(schedule)


#print('done with uncond')

meas_A = np.array([[0, 0],[0,1]])
meas_var = 0.5
#meas_var = 1e-2
meas_y = np.array([0,0.8])
#plt.savefig(str(num_particles) + '_particles.pdf')
#plt.show()

bar_alpha_end_time = np.exp(-2 * end_time)
uncond_samples = annealed_uncond_langevin(R, schedule, num_steps, num_samples=num_samples)
#rej_samples = rejection_sampler(R, schedule, num_steps, num_samples, meas_y, meas_A, meas_var)
#plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1], label='uncond_samples')
#plt.scatter(rej_samples[:, 0], rej_samples[:, 1], label='Rejection Sampling')
#plt.legend()
#true_samples = np.random.multivariate_normal(R, end_time * np.eye(2), 3000)
#plt.scatter(true_samples[:, 0], true_samples[:, 1])
#plt.show()

x, y = np.mgrid[-20:20:0.01, -20:20:0.01]
pos = np.dstack((x, y))
#print('pos:', pos.shape)
#uncond_density1 = 0.5 * multivariate_normal.pdf(pos, R, end_time * np.eye(2)) + 0.5 * multivariate_normal.pdf(pos, -R, end_time * np.eye(2))
#uncond_density2 = np.log(0.5) + scipy.special.logsumexp((multivariate_normal.logpdf(pos, R, end_time * np.eye(2)), multivariate_normal.logpdf(pos, -R, end_time * np.eye(2))))
uncond_density = np.log(0.5) + np.logaddexp(vectorized_gaussian_logpdf_single_mean(pos, R, end_time * np.eye(2)), vectorized_gaussian_logpdf_single_mean(pos, -R, end_time * np.eye(2)))
#uncond_density = np.exp(uncond_density)
#uncond_density /= np.sum(uncond_density)
#print('uncond density shape:', uncond_density.shape)
#print('densisites:', np.sum(np.exp(uncond_density)))
twisted_cond_samples = twisted_diffusion(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)
song_cond_samples = particle_filter(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)

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

plt.scatter(song_cond_samples[:, 0], song_cond_samples[:, 1], label='Song Particle Filter')
plt.scatter(twisted_cond_samples[:,0], twisted_cond_samples[:,1], label='Twisted Particle Filter')
plt.title('num particles = ' + str(num_particles))

rej_samples = rejection_sampler(R, schedule, num_steps, num_samples, meas_y, meas_A, meas_var * bar_alpha_end_time)
plt.scatter(rej_samples[:, 0], rej_samples[:, 1], label='Rejection Sampling')

plt.legend()
plt.savefig(str(num_particles) + '_particles_vectorized.pdf')
plt.show()
