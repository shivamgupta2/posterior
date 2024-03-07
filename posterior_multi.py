import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

def compute_score(x, t, R):
    #print(t, R, x)
    #print(x)
    #print(multivariate_normal.pdf(x, R, np.sqrt(t)))
    score = (-((x-R)/t) * multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, np.newaxis] * 0.5 - ((x + R)/t) * multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, np.newaxis] * 0.5)/(multivariate_normal.pdf(x, mean=R, cov=t * np.eye(2))[:, np.newaxis] * 0.5 + multivariate_normal.pdf(x, mean=-R, cov=t * np.eye(2))[:, np.newaxis] * 0.5)
    #print(score)
    return score

def create_time_schedule(num_steps, end_time, scale):
    schedule = np.zeros(num_steps)
    schedule[0] = end_time
    cur_time = end_time
    for i in range(1, num_steps):
        cur_time += cur_time * scale
        schedule[i] = cur_time
    schedule = schedule[::-1]
    #print(schedule)
    return schedule

#TODO: rewrite to take forward operator, vectorize
#edm noise schedule
def particle_filter(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=50, num_particles=10):
    dim = R.shape[0]
    cond_samples = np.zeros((num_samples, dim))
    relevant_inv = np.zeros((num_steps, dim, dim))
    for it in range(1, num_steps):
        step_size = schedule[it-1] - schedule[it]
        relevant_inv[it] = np.linalg.inv(measurement_var * np.eye(dim) + step_size * np.dot(measurement_A.T, measurement_A))

    for sample_it in range(num_samples):
        rev_schedule = schedule[::-1]
        #print(rev_schedule[0])
        noise_for_y = np.zeros((num_steps, dim))
        #noise_for_y += np.random.multivariate_normal(np.zeros(dim), rev_schedule[0] * np.eye(dim))
        noise_for_y[1:] += np.cumsum(np.random.multivariate_normal(np.zeros(dim), np.eye(dim), len(schedule)-1) * np.sqrt(rev_schedule[1:] - rev_schedule[:-1])[:, np.newaxis], axis=0)
        measured_noise_for_y = np.dot(measurement_A, noise_for_y.T).T
        noisy_y = (y + measured_noise_for_y)[::-1]
        
        #we know p(x_N | y_N) propto p(x_N) * p(y_N | x_N)
        #we also know that p(x_N) \approx N(0, schedule[0] * I_d)
        #and that p(y_N | x_N) = N(A x_N, meas_var * I_d)
        #p(x_N | y_N) is then Gaussian with mean (meas_var * I + schdule[0] * A^T A)^{-1} (schedule[0] * A^T y)
        #and covariance (schedule[0] * meas_var) * (meas_var * I + schedule[0] * A^T A)^{-1}
        x_N_cond_y_N_mean = np.dot(np.linalg.inv(measurement_var * np.eye(dim) + schedule[0] * np.dot(measurement_A.T, measurement_A)), schedule[0] * np.dot(measurement_A.T, noisy_y[0]))
        x_N_cond_y_N_cov = (schedule[0] * measurement_var) * np.linalg.inv(measurement_var * np.eye(dim) + schedule[0] * np.dot(measurement_A.T, measurement_A))
        cur_samples = np.random.multivariate_normal(x_N_cond_y_N_mean, x_N_cond_y_N_cov, num_particles)

        for it in range(1, num_steps):
            next_samples = np.zeros(cur_samples.shape)
            cur_time = schedule[it]
            step_size = schedule[it-1] - schedule[it]
            uncond_score = compute_score(cur_samples, cur_time, R)
            
            #we know that x_{k-1} | x_k, y_{k-1} is generated with prob. propto p(x_{k-1} | x_k) \cdot p(y_{k-1} | x_{k-1})
            #We have that p(x_{k-1} | x_k) \prop to N(x_k + step_size * uncond_score, step_size * I_d)
            #We have that p(y_{k-1} | x_{k-1}) \prop to N(A x_{k-1}, meas_var * I_d)
            #generate x_{k-1} | x_k, y_k-1 - it is Gaussian with mean (meas_var * I + step_size * A^T A)^{-1} * (meas_var * (x_k + step_size * uncond_score) + step_size * A^T y)
            #and covariance (step_size * meas_var) * (meas_var * I + step_size * A^T A)^{-1}
            x_N_minus_it_covar = (step_size * measurement_var) * relevant_inv[it]
            log_probs = np.zeros(num_particles)
            for i in range(num_particles):
                #print(i, it)
                x_N_minus_it_means = np.dot(relevant_inv[it], measurement_var * (cur_samples[i] + step_size * uncond_score[i]) + step_size * np.dot(measurement_A.T, noisy_y[it]))
                next_samples[i] = np.random.multivariate_normal(x_N_minus_it_means, x_N_minus_it_covar)
                # log of p(y_k | x_k) * p(x_k | x_{k+1})/p(x_k | x_{k+1}, y_k)
                log_probs[i] = multivariate_normal.logpdf(noisy_y[it], np.dot(measurement_A, next_samples[i]), measurement_var * np.eye(dim))
                log_probs[i] += multivariate_normal.logpdf(next_samples[i], cur_samples[i] + step_size * uncond_score[i], step_size * np.eye(dim))
                log_probs[i] -= multivariate_normal.logpdf(next_samples[i], x_N_minus_it_means, x_N_minus_it_covar)
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= np.sum(probs)
            sample_ids = np.random.choice(np.arange(num_particles), size=num_particles, p=probs)
            cur_samples = next_samples[sample_ids]
        #print(y, cur_samples)
        cond_samples[sample_it] = cur_samples[np.random.choice(np.arange(num_particles))]
        print('sample_it:', sample_it, 'cond_sample:', cond_samples[sample_it])
        if sample_it % 100 == 0:
            print('sample_it:', sample_it)
    return cond_samples

def annealed_uncond_langevin(R, schedule, num_steps, num_samples=50):
    start_time = schedule[0]
    end_time = schedule[len(schedule)-1]
    uncond_samples = np.random.multivariate_normal(np.zeros(2), start_time * np.eye(2), num_samples)
    print(uncond_samples.shape)
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
num_steps = 600
end_time = 0.1
schedule = create_time_schedule(num_steps, end_time, 0.01)
print(schedule)

#uncond_samples = annealed_uncond_langevin(R, schedule, num_steps)
#plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1])
#true_samples = np.random.multivariate_normal(R, end_time * np.eye(2), 3000)
#plt.scatter(true_samples[:, 0], true_samples[:, 1])
#plt.show()

#print('done with uncond')

meas_A = np.array([[0, 0], [0, 1]])
meas_var = 1e-5
meas_y = np.array([0, 1.0])
cond_samples = particle_filter(R, schedule, num_steps, meas_A, meas_var, meas_y)
plt.scatter(cond_samples[:, 0], cond_samples[:, 1])
plt.show()

x, y = np.mgrid[-5:5:0.01, -5:5:0.01]
pos = np.dstack((x, y))
uncond_density = 0.5 * multivariate_normal.pdf(pos, R, end_time * np.eye(2)) + 0.5 * multivariate_normal.pdf(pos, -R, end_time * np.eye(2))

print(pos.shape)
Ax = np.copy(pos)
Ax[:,:,0] = 0
print(Ax)
print(pos.shape, Ax.shape)
p_y_cond_x = multivariate_normal.pdf(Ax, meas_y, meas_var * np.eye(2))
p_y = 0.5 * norm.pdf(meas_y[1], R[1], end_time + meas_var) + 0.5 * norm.pdf(meas_y[1], -R[1], end_time + meas_var)
print('here:', p_y)

cond_density = uncond_density * p_y_cond_x/p_y
plt.contourf(x, y, cond_density)

print('done with cond')

plt.show()
