import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import scipy

def compute_score(x, t, R):
    score = (-((x-R)/t) * multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5 - ((x + R)/t) * multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5)/(multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5 + multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, :, np.newaxis] * 0.5)
    return score

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
    cov_inv = np.linalg.inv(covariance)
    if x.shape == means.shape:
        deviations = x - means
    else:
        deviations = x[:, np.newaxis,:] - means
    central_term = np.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations)
    return -0.5 * (constant + log_det + central_term)

#given probs of shape(num_samples, num_particles), produce res of shape(num_samples, num_particles), where each res[i, j] \in [0, num_particles) with probability probs[i, res[i,j]]
def vectorized_random_choice(probs):
    cumulative_probs = probs.cumsum(axis=1)[:, np.newaxis, :]
    unif_samples = np.random.rand(probs.shape[0], probs.shape[1])
    res = (cumulative_probs > unif_samples[:,:,np.newaxis]).argmax(axis=2)
    return res

#TODO: rewrite to take forward operator, vectorize
#edm noise schedule
def particle_filter(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=500, num_particles=1000):
    dim = R.shape[0]
    cond_samples = np.zeros((num_samples, dim))
    relevant_inv = np.zeros((num_steps, dim, dim))

    rev_schedule = schedule[::-1]
    noise_for_y = np.zeros((num_samples, num_steps, dim))
    noise_for_y[:, 1:, :] = np.cumsum(np.random.multivariate_normal(np.zeros(dim), np.eye(dim), (num_samples, len(schedule)-1)) * np.sqrt(rev_schedule[1:] - rev_schedule[:-1])[:, np.newaxis], axis=1)
    #A takes dim->dim, noise_for_y has shape (num_samples, num_steps, dim), final result should be of shape(num_samples, num_steps, dim)
    measured_noise_for_y = np.einsum('ij,klj->kli', measurement_A, noise_for_y)
    noisy_y = (y + measured_noise_for_y)[:, ::-1, :]
    print('variance:', schedule[0])
    print('seen var:', np.var(noisy_y[:, 0, 0]))

    #we know p(x_N | y_N) propto p(x_N) * p(y_N | x_N)
    #we also know that p(x_N) \approx N(0, schedule[0] * I_d)
    #and that p(y_N | x_N) = N(A x_N, meas_var * I_d)
    #p(x_N | y_N) is then Gaussian with mean (meas_var * I + schdule[0] * A^T A)^{-1} (schedule[0] * A^T y)
    #and covariance (schedule[0] * meas_var) * (meas_var * I + schedule[0] * A^T A)^{-1}
    #x_N_cond_y_N_mean has shape(num_samples, dim)
    x_N_cond_y_N_mean = np.dot(np.linalg.inv(measurement_var * np.eye(dim) + schedule[0] * np.dot(measurement_A.T, measurement_A)), schedule[0] * np.dot(measurement_A.T, noisy_y[:, 0, :].T)).T
    x_N_cond_y_N_cov = (schedule[0] * measurement_var) * np.linalg.inv(measurement_var * np.eye(dim) + schedule[0] * np.dot(measurement_A.T, measurement_A))
    #cur_samples has shape (num_samples, num_particles, dim)
    cur_samples = np.random.multivariate_normal(np.zeros(dim), x_N_cond_y_N_cov, (num_samples, num_particles)) + x_N_cond_y_N_mean[:, np.newaxis, :]
    print('here cur samples:', cur_samples[0][cur_samples[0, :, 0] < 0])
    print(noisy_y[0,0], cur_samples[0, :10])

    for it in range(1, num_steps):
        step_size = schedule[it-1] - schedule[it]
        relevant_inv[it] = np.linalg.inv(measurement_var * np.eye(dim) + step_size * np.dot(measurement_A.T, measurement_A))

        cur_time = schedule[it-1]
        uncond_score = compute_score(cur_samples, cur_time, R)

        #we know that x_{k-1} | x_k, y_{k-1} is generated with prob. propto p(x_{k-1} | x_k) \cdot p(y_{k-1} | x_{k-1})
        #We have that p(x_{k-1} | x_k) \prop to N(x_k + step_size * uncond_score, step_size * I_d)
        #We have that p(y_{k-1} | x_{k-1}) \prop to N(A x_{k-1}, meas_var * I_d)
        #generate x_{k-1} | x_k, y_k-1 - it is Gaussian with mean (meas_var * I + step_size * A^T A)^{-1} * (meas_var * (x_k + step_size * uncond_score) + step_size * A^T y)
        #and covariance (step_size * meas_var) * (meas_var * I + step_size * A^T A)^{-1}
        x_N_minus_it_covar = (step_size * measurement_var) * relevant_inv[it]
        log_probs = np.zeros((num_samples, num_particles))
        
        #x_N_minus_it_mean_helper has shape(num_samples, dim)
        x_N_minus_it_means_helper = step_size * np.dot(measurement_A.T, noisy_y[:, it, :].T).T
        x_N_minus_it_means_helper_2 = measurement_var * (cur_samples + step_size * uncond_score) + x_N_minus_it_means_helper[:, np.newaxis, :]
        #x_N_minus_it_means has shape (num_samples, num_particles, dim)
        x_N_minus_it_means = np.einsum('ij,klj->kli', relevant_inv[it], x_N_minus_it_means_helper_2)

        #next_samples has shape(num_samples, num_particles, dim) as expected
        next_samples = np.random.multivariate_normal(np.zeros(dim), x_N_minus_it_covar, (num_samples, num_particles)) + x_N_minus_it_means
        

        #resampling particles
        log_probs_term_1 = vectorized_gaussian_logpdf(noisy_y[:,it,:], np.einsum('ij,klj->kli', measurement_A, next_samples), measurement_var * np.eye(dim))
        log_probs_term_2 = vectorized_gaussian_logpdf(next_samples, cur_samples + step_size * uncond_score, step_size * np.eye(dim))
        print(log_probs_term_2[0], next_samples[0], cur_samples[0], step_size)
        log_probs_term_3 = vectorized_gaussian_logpdf(next_samples, x_N_minus_it_means, x_N_minus_it_covar)
        log_probs = log_probs_term_1 + log_probs_term_2 - log_probs_term_3
        #Finally log_probs has shape (num_samples, num_particles)
        print('log probs shape:', np.max(log_probs, axis=1).shape)

        #probs has shape(num_samples, num_particles)
        #log_probs = log_probs - np.max(log_probs, axis=1)[:, np.newaxis]
        #log_probs = log_probs - scipy.special.logsumexp(log_probs, axis=1)[:, np.newaxis]
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1)[:, np.newaxis]
        sample_ids = vectorized_random_choice(probs)
        cur_samples = next_samples[np.arange(num_samples)[:, np.newaxis], sample_ids]

    cond_sample_ids = np.random.choice(np.arange(num_particles), size=num_samples)
    print('done!!')
    print('here:', cond_sample_ids)
    cond_samples = cur_samples[np.arange(num_samples), cond_sample_ids]
    print(cond_samples[1], cur_samples[1])
    return cond_samples

def annealed_uncond_langevin(R, schedule, num_steps, num_samples=50):
    start_time = schedule[0]
    end_time = schedule[len(schedule)-1]
    uncond_samples = np.random.multivariate_normal(np.zeros(2), start_time * np.eye(2), num_samples)
    for it in range(1, num_steps):
        cur_time = schedule[it]
        step_size = schedule[it-1] - schedule[it]
        uncond_score = compute_score(uncond_samples, cur_time, R)
        uncond_samples = uncond_samples + step_size * uncond_score + np.random.multivariate_normal(np.zeros(2), step_size * np.eye(2), num_samples)
        if it % 10000 == 0:
            print('it:', it, 'time:', schedule[it])
            #plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1])
            #plt.show()
    return uncond_samples


#Parameters
R = np.ones(2)
num_steps = 500
end_time = 0.1
num_particles=200
num_samples = 500
schedule = create_time_schedule(num_steps, end_time, 0.015)
print(schedule)

#uncond_samples = annealed_uncond_langevin(R, schedule, num_steps)
#plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1])
#true_samples = np.random.multivariate_normal(R, end_time * np.eye(2), 3000)
#plt.scatter(true_samples[:, 0], true_samples[:, 1])
#plt.show()

#print('done with uncond')

meas_A = np.array([[0, 0], [0, 1]])
meas_var = 30
meas_y = np.array([0, 0])
cond_samples = particle_filter(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)
plt.scatter(cond_samples[:, 0], cond_samples[:, 1], label='Particle Filter')
plt.title('num particles = ' + str(num_particles))
#plt.savefig(str(num_particles) + '_particles.pdf')
#plt.show()

x, y = np.mgrid[-20:20:0.01, -20:20:0.01]
pos = np.dstack((x, y))
uncond_density = 0.5 * multivariate_normal.pdf(pos, R, end_time * np.eye(2)) + 0.5 * multivariate_normal.pdf(pos, -R, end_time * np.eye(2))
#uncond_density = 2 * np.log(0.5) + scipy.special.logsumexp((multivariate_normal.logpdf(pos, R, end_time * np.eye(2)), multivariate_normal.logpdf(pos, -R, end_time * np.eye(2))))

print(pos.shape)
Ax = np.copy(pos)
Ax[:,:,0] = 0
p_y_cond_x = multivariate_normal.pdf(Ax, meas_y, meas_var * np.eye(2))
print('y cond x:', p_y_cond_x.shape)
#p_y = 0.5 * norm.pdf(meas_y[1], R[1], np.sqrt(end_time + meas_var)) + 0.5 * norm.pdf(meas_y[1], -R[1], np.sqrt(end_time + meas_var))
#print('here:', p_y)

cond_density = uncond_density * p_y_cond_x
cond_density /= np.sum(cond_density)
flat_density = cond_density.flatten()
sample_index = np.random.choice(np.arange(len(x) * len(y)), p=flat_density, size=num_samples, replace=False)

selections = pos.reshape(-1, 2)[sample_index]

plt.scatter(selections[:, 0], selections[:, 1], label='True Distribution')
plt.legend()
plt.savefig(str(num_particles) + '_particles_vectorized.pdf')
plt.show()


