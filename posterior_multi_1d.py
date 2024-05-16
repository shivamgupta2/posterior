import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import logsumexp
import scipy
import torch

from mixture import GaussianMixture

def compute_score(x, t, R, weights=np.ones(2) * 0.5, variances=np.zeros(1)):
    eff_covar = t * np.eye(1)
    eff_covar[0,0] += variances[0]
    if len(x.shape) == 3:
        score = ((-((x-R)/(t + variances)) * multivariate_normal.pdf(x, mean=R, cov=eff_covar)[:, :, np.newaxis] * weights[0] -
                 ((x + R)/(t+variances)) * multivariate_normal.pdf(x, mean=-R, cov=eff_covar)[:, :, np.newaxis] * weights[1])/
                 (multivariate_normal.pdf(x, mean=R, cov=eff_covar)[:, :, np.newaxis] * weights[0] +
                  multivariate_normal.pdf(x, mean=-R, cov=eff_covar)[:, :, np.newaxis] * weights[1])
                 )
    else:
        numerator = (-((x-R)/(t+variances)) * multivariate_normal.pdf(x, mean=R, cov=eff_covar)[:, np.newaxis] * weights[0] -
                  ((x + R)/(t+variances)) * multivariate_normal.pdf(x, mean=-R, cov=eff_covar)[:, np.newaxis] * weights[1])
        denominator = (multivariate_normal.pdf(x, mean=R, cov=eff_covar)[:, np.newaxis] * weights[0] + multivariate_normal.pdf(x, mean=-R, cov=eff_covar)[:, np.newaxis] * weights[1])
        score = numerator / denominator
    return score

def compute_score_torch(x, t, R, weights=np.ones(2) * 0.5, variances=torch.zeros(2)):
    eff_covar = t * torch.eye(2)
    eff_covar[0,0] += variances[0]
    eff_covar[1,1] += variances[1]
    pos_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=R, covariance_matrix=eff_covar)
    neg_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=-R, covariance_matrix=eff_covar)
    if len(x.shape) == 3:
        score = (-((x-R)/(t + variances)) * torch.exp(pos_dist.log_prob(x))[:, :, None] * weights[0] - ((x + R)/(t + variances)) * torch.exp(neg_dist.log_prob(x))[:, :, None] * weights[1])/(torch.exp(pos_dist.log_prob(x))[:, :, None] * weights[0] + torch.exp(neg_dist.log_prob(x))[:, :, None] * weights[1])
    else:
        score = (-((x-R)/(t + variances)) * torch.exp(pos_dist.log_prob(x))[:, None] * weights[0] - ((x + R)/(t + variances)) * torch.exp(neg_dist.log_prob(x))[:, None] * weights[1])/(torch.exp(pos_dist.log_prob(x))[:, None] * weights[0] + torch.exp(neg_dist.log_prob(x))[:, None] * weights[1])
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



def create_time_schedule_eric(end_time, num_below_one, num_above):
    # In the variance preserving, would be:
    # step_size proportional to t below 1, to 1 above.
    # end_time * e^{eps num_below} = 1
    # eps num_below = log(1/end_time)

    eps = np.log(1/end_time) / num_below_one
    schedule = np.exp(eps * np.arange(num_below_one)) * end_time
    last_step_size = 1 - schedule[-1]
    later_steps = schedule[-1] + (1 + np.arange(num_above)) * last_step_size
    schedule = np.concatenate([schedule, later_steps])
    # XXX Variance exploding transforms this somehow, I'm not sure how...
    schedule = np.exp(schedule)-1
    return schedule[::-1]


#vectorized log pdfs for multiple mean vectors (given in shape (num_samples, num_particles, dim)), all with same cov, with multiple x's in shape (num_samples, dim)
#could also be that x has same shape as mean vectors
#returns log pdfs in shape(num_samples, num_particles)
def vectorized_gaussian_logpdf(x, means, covariance):
    #print(x.shape, means.shape)
    _, d = covariance.shape
    constant = d * np.log(2 * np.pi)
    _, log_det = np.linalg.slogdet(covariance)
    cov_inv = np.linalg.inv(covariance)
    if x.shape == means.shape:
        deviations = x - means
    elif len(x.shape) + 2 == len(means.shape):
        #print('other branch')
        deviations = x[np.newaxis, np.newaxis,:] - means
    else:
        deviations = x[:, np.newaxis, :] - means
    central_term = np.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations)
    return -0.5 * (constant + log_det + central_term)

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

        


def resample(X, logw, num_particles):
    bigN, d = X.shape
    num_samples = bigN // num_particles
    X = torch.tensor(X).reshape(num_samples, num_particles, d)
    logw = logw.reshape(num_samples, num_particles)
    for i in range(num_samples):
        #print(num_particles)
        #print(logw[i])
        #print(logw[i] - torch.max(logw[i]))
        w = torch.exp(logw[i] - torch.max(logw[i]))
        #if i == 0:
        #    print('WWWW', w[:10], w.max(), w.min(), torch.mean(w))
        X[i] = X[i][torch.multinomial(w, num_particles, replacement=True)]

    X = X.reshape(-1, d)
    return X

def lognormpdf(mu, sigma2):
    # XXX I think the other terms cancel anyway
    return - torch.sum(mu**2, axis=1) / (2 * sigma2)


def twisted_diffusion_eric(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=500, num_particles=1000):
    """
    Means at +R, -R
    schedule: time steps to use, sorted decreasing
    num_steps: 500, typically
    measurement_A: m x d matrix
    measurement_var: real, noise in each coordinate
    y: m long vector  [m = 2 atm]
    """
    variances = np.zeros(1) + 0.1
    y = torch.tensor(y)
    # XXX must be spherical for GaussianMixture atm
    dim = R.shape[0]

    #measurement_var = 1000
    p = GaussianMixture(dim, [variances[0]**.5]*2, [0.5,0.5], [R, -R])
    #p = GaussianMixture(dim, [1], [1], [(0,0)])
    print(f"P: {p.rs}")
    pT = p.getSmoothed(schedule[0])

    bigN = num_samples * num_particles
    # bigN x d
    x = pT.sample(bigN)
    logw = p.logptilde(schedule[0], x, y, measurement_A, measurement_var)

    #logw = logw * 0.
    #plt.ion()

    counts = []
    for i in range(1, len(schedule)):
        t = schedule[i]
        tgap = schedule[i-1] - schedule[i]

        oldx = resample(x, logw, num_particles)

        # XXX paper uses schedule[i]
        center = oldx + tgap * p.stilde(schedule[i-1], oldx, y, measurement_A, measurement_var)

        newx = center + torch.normal(0, tgap**.5, size=oldx.shape)
        
        #print(f'Variance: {torch.mean(x**2, axis=0).detach().numpy()}/{t+variances[0]} = {(torch.mean(x**2, axis=0)/(t+variances[0])).detach().numpy()}')
        
        #XXX check diffusion SDE
        logw = lognormpdf(newx - (oldx + tgap * p.getSmoothed(schedule[i-1]).score(oldx)), tgap)
        logw -= lognormpdf(newx - center, tgap)
        logw += p.logptilde(t, newx, y, measurement_A, measurement_var)
        logw -= p.logptilde(schedule[i-1], oldx, y, measurement_A, measurement_var)


        if t < .4 and False:
            import code
            d = globals()
            d.update(locals())
            code.interact(local=d)

        
        x = newx

        if (i % 50 == 0 or i in (1, len(schedule)-1)) and False:
            x0 = p.x0hat(x, t)
            topright = (x0.numpy()[:,0]).reshape(num_samples, num_particles)
            #topright = (x.numpy().dot(np.ones(2)) > 0).reshape(num_samples, num_particles)
            print('Mean x0:', np.mean(topright))
            plt.subplot(121)
            plt.plot(np.sort(np.mean(topright, axis=1)), label=f'{i}')
            counts.append(np.mean(topright))
            plt.show()

            plt.subplot(122)
            x2 = x.reshape(num_samples, num_particles, 1)
            # import code
            # d = globals()
            # d.update(locals())
            #code.interact(local=d)

        
        if i % 50 == 0 or i in (1, len(schedule)-1):
            right = torch.matmul(x, torch.ones(dim, dtype=torch.float64)) > 0
            print(f'Iteration {i} (t={t:.3f}): {torch.sum(right)/right.shape[0]:.2f} above right.  {num_samples}x{num_particles}={right.shape[0]}')
            #print(f'Variance: {torch.mean(x**2, axis=0)}')
        
    x = x.numpy()
    x = x.reshape(num_samples, num_particles, 1)
    if False:
        plt.subplot(122)
        plt.plot(counts)
        import code
        d = globals()
        d.update(locals())
        
        code.interact(local=d)
    #plt.ioff()
    #plt.scatter(x[:,0], x[:,1])
    #plt.show()
    return x#[:,0,:]

def twisted_diffusion(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=500, num_particles=1000):
    variances = np.zeros(1)
    variances[0] = 0.1
    dim = R.shape[0]
    cond_samples = np.random.normal(0, np.sqrt(schedule[0]), size=(num_samples, num_particles, dim))
    cur_time = schedule[0]
    x_0_given_x_t = cond_samples + cur_time * compute_score(cond_samples, cur_time, R, variances=variances)
    log_Tilde_p_T = vectorized_gaussian_logpdf(y, np.einsum('ij,klj->kli', measurement_A, x_0_given_x_t), measurement_var * np.eye(dim))

    log_w = log_Tilde_p_T
    
    for it in range(1, len(schedule)):
        cur_time = schedule[it]
        last_time = schedule[it-1]
        if it % 10 == 0:
            print(it, cur_time)
        step_size = schedule[it-1] - schedule[it]
        log_w -= logsumexp(log_w, axis=1)[:, np.newaxis]
        w = np.exp(log_w)
        #print('w:', np.sum(w[0]))
        resampled_indices = np.zeros((num_samples, num_particles), dtype=int)
        for i in range(num_samples):
            resampled_indices[i] = np.random.choice(num_particles, size=num_particles, p=w[i])
            cond_samples[i] = cond_samples[i][resampled_indices[i], :]
            log_Tilde_p_T[i] = log_Tilde_p_T[i][resampled_indices[i]]

        cond_samples = torch.Tensor(cond_samples)
        cond_samples = cond_samples.requires_grad_()
        uncond_score = compute_score_torch(cond_samples, last_time, torch.Tensor(R), variances=torch.Tensor(variances))
        x_0_given_x_t = cond_samples + last_time * uncond_score
        x_0_given_x_t = x_0_given_x_t.double()
        A_x_0_given_x_t = torch.einsum('ij, klj->kli', torch.Tensor(measurement_A).double(), x_0_given_x_t)
        norm_calc = -(1/(2 * measurement_var)) * torch.norm(torch.Tensor(y[None, None, :]) - A_x_0_given_x_t, dim=-1)**2
        norm_calc_sum = torch.sum(norm_calc)
        Tilde_p_T_scores = torch.autograd.grad(outputs=norm_calc_sum, inputs=cond_samples)[0]
        
        A_x_0_given_x_t = A_x_0_given_x_t.detach()
        cond_samples = cond_samples.detach()
        x_0_given_x_t = x_0_given_x_t.detach()
        uncond_score = uncond_score.detach()
        Tilde_p_T_scores = Tilde_p_T_scores.detach().numpy()

        cond_score_approx = uncond_score + Tilde_p_T_scores

        next_cond_samples = cond_samples + step_size * cond_score_approx + np.random.multivariate_normal(np.zeros(2), step_size * np.eye(2), (num_samples, num_particles))
        #next_cond_samples = newx.reshape((num_samples, num_particles, 2))

        # XXXERIC:  I think the below was wrong, x_0_given_x_t above was given cond_samples, you now want given next_cond_samples
        x_0_given_x_t_next = next_cond_samples + cur_time *compute_score_torch(next_cond_samples, cur_time, torch.Tensor(R), variances=torch.Tensor(variances))
        next_log_Tilde_p_T = vectorized_gaussian_logpdf(y, np.einsum('ij,klj->kli', measurement_A, x_0_given_x_t_next), measurement_var * np.eye(dim))
        # That is, the following is a step behind.
        # next_log_Tilde_p_T = vectorized_gaussian_logpdf(y, np.einsum('ij,klj->kli', measurement_A, x_0_given_x_t), measurement_var * np.eye(dim))
        # So in fact we could do
        # log_Tilde_p_T = vectorized_gaussian_logpdf(y, np.einsum('ij,klj->kli', measurement_A, x_0_given_x_t), measurement_var * np.eye(dim))

        
        log_w_term_1 = vectorized_gaussian_logpdf(next_cond_samples, cond_samples + step_size * uncond_score, step_size * np.eye(dim))

        #We have Pr[x_t | x_{t+1}, y] = N(x_t; x_{t+1} + \sigma^2 \grad \log p(x_{t+1} | y), sigma^2)
        #the score \grad \log p(x_{t+1} | y) is approximated as our cond_score_approx given above
        log_w_term_3 = vectorized_gaussian_logpdf(next_cond_samples, cond_samples + step_size * cond_score_approx, step_size * np.eye(dim))
        log_w = log_w_term_1 + next_log_Tilde_p_T - log_w_term_3 - log_Tilde_p_T

        cond_samples = next_cond_samples
        log_Tilde_p_T = next_log_Tilde_p_T

    return cond_samples[:,0,:] 
    
    #rev_schedule = schedule[::-1]

#TODO: rewrite to take forward operator, vectorize
#edm noise schedule
def particle_filter(R, schedule, num_steps, measurement_A, measurement_var, y, num_samples=500, num_particles=1000):
    dim = R.shape[0]
    cond_samples = np.zeros((num_samples, dim))

    rev_schedule = schedule[::-1]
    noise_for_y = np.zeros((num_samples, num_steps, dim))
    noise_for_y[:, 1:, :] = np.cumsum(np.random.multivariate_normal(np.zeros(dim), np.eye(dim), (num_samples, len(schedule)-1)) * np.sqrt(rev_schedule[1:] - rev_schedule[:-1])[:, np.newaxis], axis=1)
    #A takes dim->dim, noise_for_y has shape (num_samples, num_steps, dim), final result should be of shape(num_samples, num_steps, dim)
    measured_noise_for_y = np.einsum('ij,klj->kli', measurement_A, noise_for_y)
    noisy_y = (y + measured_noise_for_y)[:, ::-1, :]

    #we know p(x_N | y_N) propto p(x_N) * p(y_N | x_N)
    #we also know that p(x_N) \approx N(0, schedule[0] * I_d)
    #and that p(y_N | x_N) = N(A x_N, meas_var * I_d)
    #p(x_N | y_N) is then Gaussian with mean (meas_var * I + schdule[0] * A^T A)^{-1} (schedule[0] * A^T y)
    #and covariance (schedule[0] * meas_var) * (meas_var * I + schedule[0] * A^T A)^{-1}
    #x_N_cond_y_N_mean has shape(num_samples, dim)
    x_N_cond_y_N_mean = np.dot(np.linalg.inv(measurement_var * np.eye(dim) + schedule[0] * np.dot(measurement_A.T, measurement_A)), schedule[0] * np.dot(measurement_A.T, noisy_y[:, 0, :].T)).T
    x_N_cond_y_N_cov = (schedule[0] * measurement_var) * np.linalg.inv(measurement_var * np.eye(dim) + schedule[0] * np.dot(measurement_A.T, measurement_A))
    cur_samples = np.random.multivariate_normal(np.zeros(dim), x_N_cond_y_N_cov, (num_samples, num_particles)) + x_N_cond_y_N_mean[:, np.newaxis, :]

    for it in range(1, num_steps):
        step_size = schedule[it-1] - schedule[it]
        relevant_inv = np.linalg.inv(measurement_var * np.eye(dim) + step_size * np.dot(measurement_A.T, measurement_A))

        cur_time = schedule[it-1]
        uncond_score = compute_score(cur_samples, cur_time, R)

        #we know that x_{k-1} | x_k, y_{k-1} is generated with prob. propto p(x_{k-1} | x_k) \cdot p(y_{k-1} | x_{k-1})
        #We have that p(x_{k-1} | x_k) \prop to N(x_k + step_size * uncond_score, step_size * I_d)
        #We have that p(y_{k-1} | x_{k-1}) \prop to N(A x_{k-1}, meas_var * I_d)
        #generate x_{k-1} | x_k, y_k-1 - it is Gaussian with mean (meas_var * I + step_size * A^T A)^{-1} * (meas_var * (x_k + step_size * uncond_score) + step_size * A^T y)
        #and covariance (step_size * meas_var) * (meas_var * I + step_size * A^T A)^{-1}
        x_N_minus_it_covar = (step_size * measurement_var) * relevant_inv

        log_probs = np.zeros((num_samples, num_particles))
        
        #x_N_minus_it_mean_helper has shape(num_samples, dim)
        x_N_minus_it_means_helper = step_size * np.dot(measurement_A.T, noisy_y[:, it, :].T).T
        x_N_minus_it_means_helper_2 = measurement_var * (cur_samples + step_size * uncond_score) + x_N_minus_it_means_helper[:, np.newaxis, :]
        #x_N_minus_it_means has shape (num_samples, num_particles, dim)
        x_N_minus_it_means = np.einsum('ij,klj->kli', relevant_inv, x_N_minus_it_means_helper_2)

        #next_samples has shape(num_samples, num_particles, dim) as expected
        next_samples = np.random.multivariate_normal(np.zeros(dim), x_N_minus_it_covar, (num_samples, num_particles)) + x_N_minus_it_means
        

        #resampling particles
        log_probs_term_1 = vectorized_gaussian_logpdf(noisy_y[:,it,:], np.einsum('ij,klj->kli', measurement_A, next_samples), measurement_var * np.eye(dim))
        log_probs_term_2 = vectorized_gaussian_logpdf(next_samples, cur_samples + step_size * uncond_score, step_size * np.eye(dim))
        log_probs_term_3 = vectorized_gaussian_logpdf(next_samples, x_N_minus_it_means, x_N_minus_it_covar)
        log_probs = log_probs_term_1 + log_probs_term_2 - log_probs_term_3

        probs = np.exp(log_probs - np.max(log_probs, axis=1)[:, np.newaxis])
        probs /= np.sum(probs, axis=1)[:, np.newaxis]

        resampled_indices = np.zeros((num_samples, num_particles), dtype=int)
        for i in range(num_samples):
            resampled_indices[i] = np.random.choice(num_particles, size=num_particles, p=probs[i])
            cur_samples[i] = next_samples[i][resampled_indices[i], :]

        #cur_samples = next_samples[np.arange(num_samples)[:, np.newaxis], resampled_indices]
        #cur_samples = next_samples

        #if it % 100 == 0:
            #print('it:', it, 'time:', schedule[it])
            #plt.scatter(next_samples[:, :, 0], next_samples[:, :, 1])
            #plt.show()
    cond_samples = cur_samples[:, 0, :]
    #cond_sample_ids = np.random.choice(np.arange(num_particles), size=num_samples)
    #print('done!!')
    #cond_samples = cur_samples[np.arange(num_samples), cond_sample_ids]
    #print(cond_samples[1], cur_samples[1])
    return cond_samples

def annealed_uncond_langevin(R, schedule, num_samples=50):
    num_steps = len(schedule)
    variances = np.zeros(1)
    variances[0] = 0.1
    start_time = schedule[0]
    end_time = schedule[-1]
    uncond_samples = np.random.multivariate_normal(np.zeros(1), start_time * np.eye(1), num_samples)
    for it in range(1, num_steps):
        cur_time = schedule[it]
        step_size = schedule[it-1] - schedule[it]


        uncond_score = compute_score(uncond_samples, cur_time, R, variances=variances)

        #print(step_size)
        # Check score calculation is the same, and it is up to 1e-6 error
        # p = GaussianMixture(2, [variances[0]**.5]*2, [0.5,0.5], [R, -R])
        # uncond_score2 = p.getSmoothed(cur_time).score(uncond_samples)
        # print(cur_time, ':')
        # print(np.linalg.norm(uncond_score - uncond_score2.numpy()))
        
        uncond_samples = uncond_samples + step_size * uncond_score + np.random.multivariate_normal(np.zeros(1), step_size * np.eye(1), num_samples)
        if it % 100000 == 0:
            print('it:', it, 'time:', schedule[it])
            plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1])
            p = GaussianMixture(2, [variances[0]**.5]*2, [0.5,0.5], [R, -R])
            samples2 = p.getSmoothed(cur_time).sample(num_samples)
            plt.scatter(samples2[:, 0], samples2[:, 1])
            plt.show()

            
    return uncond_samples

def rejection_sampler(R, schedule, num_steps, num_samples, y, meas_A, meas_var):
    K = 2*num_samples
    samplesets = []
    while sum(map(len, samplesets)) < num_samples:
        uncond_samples = annealed_uncond_langevin(R, schedule, num_samples)
        Ax = np.dot(meas_A, uncond_samples.T).T
        accept_prob = np.exp(-np.linalg.norm(Ax - y, axis=1)**2/(2 * meas_var))
        unif_samples = np.random.rand(num_samples)
        samplesets.append(uncond_samples[unif_samples < accept_prob])
        #cond_samples[unif_samples < accept_prob] = uncond_samples[unif_samples < accept_prob]
        #done[unif_samples < accept_prob] = np.ones(num_samples, dtype=int)[unif_samples < accept_prob]
        print(list(map(len, samplesets)))
    cond_samples = np.vstack(samplesets)[:num_samples]
    return cond_samples


def vectorized_gaussian_logpdf_single_mean(x, mean, covariance):
    _, d = covariance.shape
    constant = d * np.log(2 * np.pi)
    _, log_det = np.linalg.slogdet(covariance)
    cov_inv = np.linalg.inv(covariance)
    deviations = x - mean
    central_term = np.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations)
    print('here:', central_term.shape)
    return -0.5 * central_term

#Parameters
R = np.ones(2)
R = np.array([1.])
num_steps = 50
end_time = 0.0001
num_particles = 100
num_samples = 5000
schedule = create_time_schedule_eric(end_time, num_steps, num_steps)
#schedule = create_time_schedule(num_steps, end_time, 0.025)
#print(schedule)
#plt.plot(schedule)
#plt.show()

uncond_samples = annealed_uncond_langevin(R, schedule, 5000)
#plt.scatter(uncond_samples[:, 0], uncond_samples[:, 1])
#true_samples = np.random.multivariate_normal(R, end_time * np.eye(2), 3000)
# plt.scatter(true_samples[:, 0], true_samples[:, 1])
# ax = plt.gca()
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])
# plt.show()

#print('done with uncond')

meas_A = np.array([[1]])
meas_var = 0.1
#meas_var = 1e-2
meas_y = np.array([0.2])
#plt.savefig(str(num_particles) + '_particles.pdf')
#plt.show()

#x, y = np.mgrid[-20:20:0.01, -20:20:0.01]
#pos = np.dstack((x, y))
#print('pos:', pos.shape)
#uncond_density1 = 0.5 * multivariate_normal.pdf(pos, R, end_time * np.eye(2)) + 0.5 * multivariate_normal.pdf(pos, -R, end_time * np.eye(2))
#uncond_density2 = np.log(0.5) + scipy.special.logsumexp((multivariate_normal.logpdf(pos, R, end_time * np.eye(2)), multivariate_normal.logpdf(pos, -R, end_time * np.eye(2))))
#uncond_density = np.log(0.5) + np.logaddexp(vectorized_gaussian_logpdf_single_mean(pos, R, end_time * np.eye(2)), vectorized_gaussian_logpdf_single_mean(pos, -R, end_time * np.eye(2)))
#uncond_density = np.exp(uncond_density)
#uncond_density /= np.sum(uncond_density)
#print('uncond density shape:', uncond_density.shape)
#print('densisites:', np.sum(np.exp(uncond_density)))
#cond_samples = particle_filter(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)
cond_samples = twisted_diffusion_eric(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)
#cond_samples2 = twisted_diffusion(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)



#song_cond_samples = particle_filter(R, schedule, num_steps, meas_A, meas_var, meas_y, num_samples=num_samples, num_particles=num_particles)
#song_cond_samples = particle_filter(

# Ax = np.copy(pos)
# Ax[:,:,0] = 0
# #p_y_cond_x = multivariate_normal.logpdf(Ax, meas_y, meas_var * np.eye(2))
# p_y_cond_x = vectorized_gaussian_logpdf_single_mean(Ax, meas_y, meas_var * np.eye(2))
# #print('p_y_cond_x_shape:', p_y_cond_x.shape)
# #p_y = 0.5 * norm.pdf(meas_y[1], R[1], np.sqrt(end_time + meas_var)) + 0.5 * norm.pdf(meas_y[1], -R[1], np.sqrt(end_time + meas_var))
# #print('here:', p_y)

#cond_density = uncond_density * p_y_cond_x
# cond_density = np.exp(uncond_density + p_y_cond_x)
# cond_density /= np.sum(cond_density)
# flat_density = cond_density.flatten()
#flat_uncond_density = uncond_density.flatten()
# sample_index = np.random.choice(np.arange(len(x) * len(y)), p=flat_density, size=num_samples, replace=False)

#plt.scatter(cond_samples[:, 0, 0], cond_samples[:, 0, 0], label='Twisted Diffusion Particle Filter (Eric)')
#plt.scatter(cond_samples2[:, 0], cond_samples2[:, 1], label='Twisted Diffusion Particle Filter (Shivam)')

rej_samples = rejection_sampler(R, schedule, num_steps, 20000, meas_y, meas_A, meas_var)
#plt.scatter(rej_samples[:, 0], rej_samples[:, 0], label='Rejection Sampling')
plt.hist(cond_samples.flatten(), bins=20, density=True, histtype='step')
plt.hist(rej_samples, bins=20, density=True, histtype='step')

cond_samples = np.array(cond_samples)
frac_rej = np.mean(rej_samples.dot([1]) < 0)
frac_cond = np.mean(cond_samples.dot([1]) < 0)
text=f'num particles = {num_particles}, num bottom left: {frac_cond:0.3f} conditional, {frac_rej:0.3f} rejection'
print(text) 
plt.title(text)

plt.legend()
ax = plt.gca()
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
plt.savefig(str(num_particles) + '_particles_vectorized.pdf')
plt.show()


plt.ion()
plt.clf()
rej_samples = rej_samples[:,0]
cond_samples = cond_samples[:,:,0]
print(f'Cond: {np.mean(cond_samples > 0):.2f} Rej: {np.mean(rej_samples > 0):.2f}')
plt.hist(cond_samples.flatten(), bins=20, density=True, histtype='step')
plt.hist(rej_samples, bins=20, density=True, histtype='step')
import code
d = globals()
d.update(locals())

code.interact(local=d)
