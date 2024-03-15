import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_score(x, t, R):
    return (-((x-R)/t) * norm.pdf(x, R, np.sqrt(t)) * 0.5 - ((x + R)/t) * norm.pdf(x, -R, np.sqrt(t)) * 0.5)/(norm.pdf(x, R, np.sqrt(t)) * 0.5 + norm.pdf(x, -R, np.sqrt(t)) * 0.5)

def compute_conditional_score(x, y, t, R, eta_var):
#    y_conditioned_on_x_t_score = norm.pdf(y, x, np.sqrt(eta))
    x_t_score = compute_score(x, t, R)
    y_conditioned_on_x_t_score = (y - x)/eta_var
    #print(x_t_score, y_conditioned_on_x_t_score)
    return x_t_score + y_conditioned_on_x_t_score

def plot_density(t, R):
    x = np.linspace(-7, 7, num=10000)
    y = norm.pdf(x, -R, np.sqrt(t)) * 0.5 + norm.pdf(x, R, np.sqrt(t)) * 0.5
    plt.plot(x, y)

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

def plot_samples(samples, label):
    #Plotting the results
    x = np.linspace(-7, 7, num=10000)
    #plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=100, density=True, alpha=0.6, label=label)
    plt.title("Samples from a Mixture of Two Gaussians using Annealed Langevin Dynamics")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    #plot_density(1, R)

def annealed_langevin_dynamics(R, schedule, y=0, eta_var=1, num_samples = 3000):
    start_time = schedule[0]
    end_time = schedule[len(schedule)-1]
    increasing_schedule = schedule[::-1]
    #print('start time:', start_time)
    noisy_y = norm.rvs(scale=1, size=(num_samples, len(schedule) - 1)) * np.sqrt(increasing_schedule[1:] - increasing_schedule[:-1])
    noisy_y = np.cumsum(noisy_y, axis=1)
    #print('noise samples:', np.std(np.cumsum(noisy_y, axis=1), axis=0))
    noisy_y = np.flip(noisy_y, axis=1) + y
    print('noise samples:', np.std(noisy_y[:,0]))
    #noisy_y = np.cumsum(norm.rvs(scale=1, size=(num_samples, len(schedule)-1)) * np.sqrt(increasing_schedule[1:] - increasing_schedule[:-1]), axis=1) + y
    print('noisy_y_shape:', noisy_y.shape)
    print('noisy_y:', noisy_y[:,0], np.std(noisy_y[:,0]))
    x = np.linspace(-7, 7, num=10000)
    end_uncond_density = 0.5 * norm.pdf(x, R, np.sqrt(end_time)) + 0.5 * norm.pdf(x, -R, np.sqrt(end_time))
    end_cond_density = (end_uncond_density * norm.pdf(y, x, np.sqrt(eta_var)))/(0.5 * norm.pdf(y, R, np.sqrt(end_time + eta_var)) + 0.5 * norm.pdf(y, -R, np.sqrt(end_time + eta_var)))
    score = compute_score(x, 1, R)
    
    #derivative = (density[1:] - density[:-1])/(x[1:] - x[:-1])
    #score_f = derivative/density[:-1]

    """Sample from a mixture of two Gaussians using annealed Langevin dynamics."""
    #samples = np.random.normal(0, np.sqrt(schedule[0]), num_samples)
    uncond_samples = np.random.normal(0, np.sqrt(schedule[0]), num_samples)
    cond_samples = np.random.normal(0, np.sqrt(schedule[0]), num_samples)
    cond_samples_noisy = np.random.normal(0, np.sqrt(schedule[0]), num_samples)
    num_times = 10000
    step_size = schedule[0] - schedule[1]
    for i in range(num_times):
        cond_score = compute_conditional_score(cond_samples, y, schedule[0], R, eta_var)
        cond_samples = cond_samples + step_size * cond_score/2 + np.random.normal(0, np.sqrt(step_size), num_samples)
        cond_score_noisy = compute_conditional_score(cond_samples_noisy, noisy_y[:,0], schedule[0], R, eta_var)
        cond_samples_noisy = cond_samples_noisy + step_size * cond_score_noisy/2 + np.random.normal(0, np.sqrt(step_size), num_samples)
    uncond_density = 0.5 * norm.pdf(x, R, np.sqrt(schedule[0])) + 0.5 * norm.pdf(x, -R, np.sqrt(schedule[0]))
    cond_density = (uncond_density * norm.pdf(y, x, np.sqrt(eta_var)))/(0.5 * norm.pdf(y, R, np.sqrt(schedule[0] + eta_var)) + 0.5 * norm.pdf(y, -R, np.sqrt(schedule[0]+eta_var)))
    plt.plot(x, uncond_density, label='Unconditional Density')
    plt.plot(x, cond_density, label='Conditional Density')
    plot_samples(uncond_samples, 'Unconditional Samples')
    plot_samples(cond_samples, 'Conditional Samples')
    plot_samples(cond_samples_noisy, 'Conditional Samples for Noisy y')
    plt.show()

    #step_size = (start_time - end_time)/n_iterations
    num_times = 10000
    for it in range(1, n_iterations):
        cur_time = schedule[it]
        step_size = schedule[it-1]-schedule[it]
        uncond_score = compute_score(uncond_samples, cur_time, R)
        uncond_samples = uncond_samples + step_size * uncond_score + np.random.normal(0, np.sqrt(step_size), num_samples)
        cond_score = compute_conditional_score(cond_samples, y, cur_time, R, eta_var)
        cond_samples = cond_samples + (step_size) * cond_score/2 + np.random.normal(0, np.sqrt(step_size), num_samples)
        cond_score_noisy = compute_conditional_score(cond_samples_noisy, noisy_y[:,it], cur_time, R, eta_var)
        cond_samples_noisy = cond_samples_noisy + step_size * cond_score_noisy/2 + np.random.normal(0, np.sqrt(step_size), num_samples)
        #print(samples[:10], score[:10], step_size)
        if it % 4999 == 0 or (it > 50000 and it % 999 == 0):
            uncond_density = 0.5 * norm.pdf(x, R, np.sqrt(schedule[it])) + 0.5 * norm.pdf(x, -R, np.sqrt(schedule[it]))
            cond_density = (uncond_density * norm.pdf(y, x, np.sqrt(eta_var)))/(0.5 * norm.pdf(y, R, np.sqrt(schedule[it] + eta_var)) + 0.5 * norm.pdf(y, -R, np.sqrt(schedule[it]+eta_var)))
            cond_derivative = (cond_density[1:] - cond_density[:-1])/(x[1:] - x[:-1])
            est_cond_score = cond_derivative/cond_density[:-1]
            cond_score = compute_conditional_score(x, y, cur_time, R, 1)
            plt.plot(x[:-1], est_cond_score, label='est cond score')
            plt.plot(x, cond_score, label='cond score')
            plt.legend()
            #plt.show()
            plt.clf()
            for j in range(num_times):
                cond_score = compute_conditional_score(cond_samples, y, cur_time, R, 1)
                cond_samples = cond_samples + (step_size) * cond_score/2 + np.random.normal(0, np.sqrt(step_size), num_samples)
                cond_score_noisy = compute_conditional_score(cond_samples_noisy, noisy_y[:,it], cur_time, R, eta_var)
                cond_samples_noisy = cond_samples_noisy + step_size * cond_score_noisy/2 + np.random.normal(0, np.sqrt(step_size), num_samples)
            #print('cond_density_sum:', np.sum(cond_density * (x[1] - x[0])))
            plt.plot(x, uncond_density, label='Unconditional Density')
            plt.plot(x, cond_density, label='Conditional Density')
            plt.plot(x, end_cond_density, label='End Conditional Density')
            plot_samples(uncond_samples, 'Unconditional Samples')
            plot_samples(cond_samples, 'Conditional Samples')
            plot_samples(cond_samples_noisy, 'Conditional Samples for Noisy y')
            #plot_samples(noisy_y[:,it], 'Noisy y')
            plt.show()
            plt.savefig('it:' + str(it) + 'samples and dist')
            plt.clf()
        print('cond_prob that x<0:', np.sum(cond_density[x < 0] * (x[1] - x[0])))
        print('noisy_y_var:', np.std(noisy_y[:,it]) ** 2)
        print(cur_time)

    return cond_samples

# Parameters
R = 3  # Distance of the means of the Gaussians from the origin
n_iterations = 60000
end_time = 0.1
schedule = create_time_schedule(n_iterations, end_time, 0.0001)


# Run the annealed Langevin dynamics
#samples = annealed_langevin_dynamics(R, schedule)
samples = annealed_langevin_dynamics(R, schedule, y=0.1)
