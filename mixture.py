import numpy as np
import torch

from scipy import stats


class GaussianMixture:
    def __init__(self, d, rs, ps=None, centers=None):
        self.d = d
        self.k = len(rs)
        if centers is None:
            centers = np.random.randn(self.k, d) / d**.5
        self.centers = torch.tensor(centers)
        self.rs = torch.tensor(rs, dtype=torch.float64)
        self.Iests = {}
        if ps is None:
            ps = np.array([1]*self.k)
        assert len(ps) == len(rs)
        self.ps = np.array(ps) / np.sum(ps)


    def sample(self, n):
        samples = np.random.randn(n, self.d)
        mix = stats.rv_discrete(values=(np.arange(self.k), self.ps))
        choices = mix.rvs(size=n)
        result = self.centers[choices] + self.rs[choices].reshape([-1, 1]) * samples
        return result

    def pmf(self, xs):
        assert xs.shape[-1] == self.d
        xs = xs.reshape((-1, 1, self.d))
        offsets = xs - self.centers  #n x k x d
        pdf = np.zeros(len(xs))
        for i in range(self.k):
            pdf += stats.multivariate_normal.pdf(offsets[:,i,:], cov=torch.eye(self.d)*self.rs[i])
        return pdf

    def score(self, xs):
        """
        xs: n x d
        """
        #xs = torch.tensor(xs)
        #print("SCORE", xs.shape)
        if len(xs.shape) == 1:
            xs = xs.reshape((1, -1))
        n = len(xs)
        # Single gaussian: -(Sigma^{-1})(x-mu)
        assert xs.shape[-1] == self.d
        xs = xs.reshape((-1, 1, self.d))
        offsets = xs - self.centers  #n x k x d
        #print('OFFSETS', offsets)

        # Compute weighted average of individual scores. (sum p)'/(sum p) = (sum (p'/p) * p)/sum p
        # Use logpdfs for better conditioning (shift so max prob. is constant)
        logpdfs = []
        scores = []
        for i in range(self.k):
            #logpdf = stats.multivariate_normal.logpdf(offsets[:,i,:], cov=torch.eye(self.d)*self.rs[i]**2)
            dist = torch.distributions.multivariate_normal.MultivariateNormal(self.centers[i], torch.eye(self.d)*self.rs[i]**2)
            logpdf = dist.log_prob(xs[:,0,:])
            logpdfs.append(logpdf)
            #print('PDFS', logpdfs)
            scores.append( - self.rs[i]**(-2) * offsets[:,i,:])
        logpdfs = torch.stack(logpdfs) #k x n
        logpdfs -= torch.max(logpdfs, axis=0).values
        pdfsa = torch.exp(logpdfs).reshape(self.k, n, 1)
        pdfs = pdfsa / torch.sum(pdfsa, axis=0)
        scores = torch.stack(scores) # k x n x d
        score = torch.sum(pdfs * scores, axis=0)
        #print('HRM', logpdfs.shape, scores.shape, pdfs.shape, score.shape, (pdfs * scores).shape)
        #print('ZZZ', xs.shape)
        #torch.autograd.grad(xs[0,0,0], xs)
        return score

    
    def mean(self):
        return np.sum([c * p for c, p in zip(self.centers, self.ps)], axis=0)

    def var(self):
        mu = self.mean()
        Sigma = sum(p * ((c-mu).reshape([-1,1])*(c-mu).reshape([1,-1]) + torch.eye(self.d)*r) for p, c, r in zip(self.ps, self.centers, self.rs))
        return Sigma

    def getI(self, N):
        maxbatch = max(1, 10**6 // self.d**2)
        ans = 0
        n = N
        while n > 0:
            batch = min(n, maxbatch)
            n -= batch
            # batch samples from each gaussian
            samples = np.random.randn(batch, 1, self.d) * self.rs.reshape([1, -1, 1]) + self.centers.reshape([1, -1, self.d])
            # batch x k x d
            for i in range(self.k):
                scores = self.score(samples[:,i,:]) #batch x d
                SST = scores.reshape((batch, -1, 1)) * scores.reshape((batch, 1, -1))
                ans += np.sum(SST, axis=0) * self.ps[i]
        return ans / N

    def getI_old(self, N):
        maxbatch = max(1, 10**6 // self.d**2)
        ans = 0
        n = N
        while n > 0:
            batch = min(n, maxbatch)
            n -= batch
            samples = self.sample(batch)
            scores = self.score(samples) # batch x d
            SST = scores.reshape((batch, -1, 1)) * scores.reshape((batch, 1, -1))
            ans += np.sum(SST, axis=0)
        return ans / N


    def getSmoothed(self, r2):
        return GaussianMixture(self.d, np.sqrt(self.rs**2 + r2), self.ps, self.centers)

    def x0hat(self, xs, t):
        # Apply this to the t-smoothed mixture
        return xs + self.getSmoothed(t).score(xs) * t

    def logptilde(self, t, xs, y, measurement_A, measurement_var):
        x0 = self.x0hat(xs, t)
        #print('ptilde says:', xs, pt.score(xs), x0)
        measurement_A = torch.tensor(measurement_A, dtype=torch.float64)
        #print(measurement_A.shape)
        residual = y - torch.matmul(measurement_A, x0.T).T
        return - torch.sum(residual**2/ (2*measurement_var), axis=1)

    def stilde(self, t, xs, y, measurement_A, measurement_var):
        torch.autograd.set_detect_anomaly(True)
        xs = torch.tensor(xs)
        xs.requires_grad_()
        pt = self.getSmoothed(t)
        score = pt.score(xs)
        fn = lambda x: self.logptilde(t, x, y, measurement_A, measurement_var)
        #print('STARTING stilde', xs, len(xs))
        # for i in range(len(xs)):
        #     g = torch.autograd.grad(fn(xs[i]), xs)
        #     print('ZZZ', g)
        #     score += g[0]
        g = torch.autograd.grad(torch.sum(fn(xs)), xs)
        #print('grad', g)
        score += g[0]
        xs.detach()
        return score

    def ptildedist(self, t, tgap, xtplus, y, measurement_A, measurement_var):
        center = xtplus + t * self.stilde(t, xs, y, measurement_A, measurement_var)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(center, tgap)
        return dist

    def getsmoothedI(self, N, r):
        m2 = GaussianMixture(self.d, np.sqrt(self.rs**2 + r**2), self.ps, self.centers)
        return m2.getI(N)

    def runalg(self, samples, r, Iest=None, steps=1):
        samples = np.array(samples)
        n = samples.shape[0]
        m2 = GaussianMixture(self.d, np.sqrt(self.rs**2 + r**2), self.ps, self.centers)
        smoothsamples = samples + np.random.randn(n, self.d)*r

        if Iest is None:
            Iest = self.Iests.get(r, None)
        if Iest is None:
            print(f"Getting I for {r}...")
            Iest = m2.getI(10**6)
            self.Iests[r] = Iest
            print("...done")

        mean = np.mean(samples, axis=0) - self.mean()
        for i in range(steps):
            meanscore = np.mean(m2.score(smoothsamples - mean), axis=0)
            shift = np.linalg.inv(Iest).dot(meanscore)
            mean = mean - shift
        #print(f"Run {r} {steps} {n}: Initial {np.linalg.norm(np.mean(samples, axis=0) - self.mean())}, final {np.linalg.norm(mean, 2)}")
        return mean


    def repeatedMLE(self, samples, r, Iest, count=100):
        ans = []
        for i in range(count):
            ans.append(self.runalg(samples, r, Iest))
        return np.mean(ans, axis=0)

    def test_recovery(self, samples, r):
        ans = {}
        #ans['avgsmoothedMLE'] = np.linalg.norm(self.repeatedMLE(samples, r, Ir))
        scale = len(samples)**.5  #scale up to C
        ans['Ours, r=.1'] = np.linalg.norm(self.runalg(samples, .1)) *scale
        ans['Ours, r=.01'] = np.linalg.norm(self.runalg(samples, .01)) *scale
        ans['Ours, r=.001'] = np.linalg.norm(self.runalg(samples, .001)) *scale
        ans['Ours, r=0'] = np.linalg.norm(self.runalg(samples, 0)) *scale

        ans['Ours-10, r=.1'] = np.linalg.norm(self.runalg(samples, .1,steps=10)) *scale
        ans['Ours-10, r=.01'] = np.linalg.norm(self.runalg(samples, .01,steps=10)) *scale
        ans['Ours-10, r=.001'] = np.linalg.norm(self.runalg(samples, .001,steps=10)) *scale
        ans['Ours-10, r=0'] = np.linalg.norm(self.runalg(samples, 0,steps=10)) *scale
        # ans['smoothedMLE-10'] = np.linalg.norm(self.runalg(samples, r, None,10)) *scale
        # ans['MLE'] = np.linalg.norm(self.runalg(samples, 0, I))*scale
        # ans['MLE-10'] = np.linalg.norm(self.runalg(samples, 0, I,10))*scale
        ans['mean'] = np.linalg.norm(np.mean(samples, axis=0)-self.mean())*scale
        #print(ans)
        return ans

    def test_recovery_repeat(self, N, k, r):
        ans = {}
        for i in range(k):
            result = self.test_recovery(self.sample(N), r)
            for key in result:
                ans.setdefault(key, []).append(result[key])
        for key in ans:
            ans[key].sort()
        return ans


if __name__ == '__main__':
    #Want r ~ sqrt(Var) (d/n)^{1/4}
    # m = GaussianMixture(d, [1, 3, 0.001], [1, 1, 0.0001], [[-1,0,0],[1,0,0], [0,9999,0]])
    # m.getI(10**6)
    # samples = m.samples(1000)
    # m.run(samples, 0.1)

    # reload(main);np.random.seed(17);m = main.GaussianMixture(10, [1, 3, 0.001])
    # Ir = m.getsmoothedI(10**7, 0.1)
    # np.median([np.linalg.norm(np.mean(m.sample(10000), axis=0) - m.mean()) for _ in range(100)])
    # np.median([np.linalg.norm(m.runalg(m.sample(1000), 0.1, Ir)) for _ in range(100)])


    # reload(main);np.random.seed(17);r=0.1;m = main.GaussianMixture(10, [1, 3, 1e-2], [1, 1, 0.0001], [[-1]+[0]*9,[1]+[0]*9, [999]*10])
    # I = m.getI(10**7);Ir = m.getsmoothedI(10**7, r)
    # m.test_recovery(m.sample(100000), r, I, Ir)

    r = 0.1
    d = 20
    #m = GaussianMixture(d, [1, 3, 1e-2], [1, 1, 0.0001], [[-1]+[0]*(d-1),[1]+[0]*(d-1), [999]*d])
    m = GaussianMixture(d, [1, 3, 1e-3], [1, 1, 0.0001], [[-1]+[0]*(d-1),[1]+[0]*(d-1), [0,10000]+[0]*(d-2)])
    #m = GaussianMixture(d, [1, 3], [1, 1], [[-1]+[0]*(d-1),[1]+[0]*(d-1), ])
    print(f"Running: {m.rs} {m.ps}")
    #print("Computing I...")
    #I = m.getI(10**7)
    #print("Computing Ir...")
    #Ir = m.getsmoothedI(10**7, r)
    #print('sqrt trace:', np.sqrt(np.trace(np.linalg.inv(I))), np.sqrt(np.trace(np.linalg.inv(Ir))))
    #print("done")
    results = {}
    repetitions = 100
    for i in range(1, 5):
        ans = m.test_recovery_repeat(10**i, repetitions, r)
        results[i] = ans
        print(f"10**{i}")
        for key in ans:
            print(key, np.median(ans[key]), ans[key][(len(ans[key])*9)//10], np.mean(ans[key]), np.std(ans[key]), np.max(ans[key]))

    print()
    print("Scaled errors [L2 error is C/sqrt(n) for the given C]")
    print('sqrt trace:')
    for r in m.Iests:
        print(r, np.sqrt(np.trace(np.linalg.inv(m.Iests[r]))))

    print(f"Reminder: {m.rs} {m.ps}")
    print()
    for (label, func) in [('Medians:', np.median),
                          ('90th percentile', lambda x: x[(9*len(x))//10]),
                          ('Means:', np.mean),
                          ('Maxes:', np.max),
                          ]:
        print(label)
        print(f'{"N":11}\t' + '\t'.join(f'10^{k}' for k in results.keys()))
        for key in ans:
            vals = [func(results[i][key]) for i in results]
            line = f'{key:11}\t| ' + ' | '.join(f'{val:0.2f}' for val in vals)
            print(line)
        print()


'''

Simulations
-----------

The reviewers have requested numerical simulations, so we give them
here.  We will present plots in the final version of the paper, but a
table must suffice for the rebuttal.

We consider a mixture of three gaussians, two "normal" and one very
narrow and rare: $d = 20$, and

 $x \sim \mu + N(-e_1, I) + N(e_1, 9I) + 10^{-4} N(10^4 e_2, 10^{-8} I)$.

We consider three algorithms: our algorithm with smoothing radius 0.1;
the empirical mean; and an approximation to the MLE given by Newton's
method (i.e., our algorithm except with r=0 and multiple steps).

For each algorithm, and for a variety of sample sizes $N$, we compute
$\sqrt{N}$ times the estimation L2 error.  Our theorem suggests that
this should be about $\sqrt{\Tr(I_r^{-1})} \approx 6.0$ for our
algorithm with r=0.1, which is significantly better than the mean's
typical error of $\sqrt{\Tr(\Sigma)} \approx 70$, but significantly
worse than the Cramer-Rao bound $\sqrt{\Tr(I^{-1})} \approx 0.06$.

In our experiments, the median error is as follows:

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours    	6.10	6.07	6.19	5.98	5.82	6.07
no-smoothing   	9.69	11.28	18.74	50.97	0.07	0.06
mean       	9.70	11.29	18.76	51.01	61.16	50.58

while the 90th percentile error is as follows:

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours    	8.36	7.25	7.67	7.18	7.20	7.30
no-smoothing   	12.22	13.66	21.75	147.04	126.23	0.08
mean       	12.23	13.67	21.77	147.16	126.31	121.19

[XXX: not sure how to handle MLE--- don't know how to compute it!  Can
run several newton steps for r=0, but may well not converge;
empirically it works better for radius 1e-2 and worse for radius 1e-4.]

----

N(-e_1, I) + N(e_1, 9I) + 10^{-4} N(10^4 e_2, 10^{-8}I)

Run the algorithm with r=0.1, use all the samples for initial estimate
[not in paper but we know now that it's fine], use 10^7 samples to
estimate I_r.



Scaled errors [L2 error is C/sqrt(n) for the given C]
sqrt trace:
0.1 6.0035185517353655
0.01 4.354010284264186
0.001 0.6320791749120275
0 0.06324206952079307
Reminder: [1.e+00 3.e+00 1.e-04] [4.99975001e-01 4.99975001e-01 4.99975001e-05]

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	6.20	6.00	6.13	5.89	5.77	5.89
Ours, r=.01	7.06	7.23	10.38	24.53	7.00	6.25
Ours, r=.001	10.22	10.83	18.75	50.05	7.54	9.59
Ours, r=0  	10.29	10.93	18.93	50.63	7.47	9.70
Ours-10, r=.1	6.04	5.83	5.89	5.76	5.61	5.90
Ours-10, r=.01	6.00	5.79	6.00	5.55	4.25	4.20
Ours-10, r=.001	9.65	10.02	17.23	45.36	0.67	0.61
Ours-10, r=0	10.28	10.92	18.91	50.58	0.07	0.06
mean       	10.29	10.93	18.93	50.63	37.33	69.39

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	8.62	7.25	8.05	7.31	7.54	7.09
Ours, r=.01	10.30	8.77	13.43	29.38	47.87	15.97
Ours, r=.001	12.72	13.24	23.28	56.34	100.06	32.39
Ours, r=0  	12.77	13.33	23.49	56.91	126.37	32.81
Ours-10, r=.1	7.75	7.23	7.52	7.21	7.18	6.90
Ours-10, r=.01	7.65	7.24	8.05	8.89	5.49	5.19
Ours-10, r=.001	12.13	12.31	21.39	51.37	12.24	0.76
Ours-10, r=0	12.77	13.32	23.48	56.86	126.25	0.08
mean       	12.77	13.33	23.50	56.92	126.38	126.26

Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	6.60	5.94	17.00	6.00	5.87	5.92
Ours, r=.01	7.45	7.26	165.54	41.61	16.90	8.34
Ours, r=.001	10.08	10.82	35.77	85.92	31.55	13.20
Ours, r=0  	10.15	10.91	35.95	59.57	31.99	13.27
Ours-10, r=.1	6.16	5.89	5.98	5.90	5.80	5.89
Ours-10, r=.01	6.14	5.86	477.36	74.14	4.40	4.27
Ours-10, r=.001	9.52	10.09	34.30	78.16	28.10	0.63
Ours-10, r=0	10.14	10.90	35.93	59.52	15.31	0.06
mean       	10.15	10.91	35.95	59.58	56.61	70.98

Maxes:
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	13.08	8.09	190.50	9.13	9.24	8.26
Ours, r=.01	13.16	9.73	2616.65	222.64	108.96	31.83
Ours, r=.001	15.28	14.81	304.81	460.02	226.66	65.08
Ours, r=0  	15.36	14.94	304.78	154.86	190.43	65.90
Ours-10, r=.1	12.80	8.34	8.59	8.82	8.83	8.00
Ours-10, r=.01	12.83	8.17	7921.73	773.69	6.14	6.07
Ours-10, r=.001	14.52	13.84	305.11	419.48	1068.93	0.99
Ours-10, r=0	15.35	14.92	304.78	154.75	190.31	0.09
mean       	15.36	14.94	304.78	154.88	190.45	183.07



Reminder: [1.e+00 3.e+00 1.e-04] [4.99975001e-01 4.99975001e-01 4.99975001e-05]
Scaled errors [L2 error is C/sqrt(n) for the given C]
sqrt trace:
0.1 6.0034836081649505
0.01 4.354115891980285
0.001 0.6320782950868129
0 0.06325062542591088

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	6.62	6.12	6.02	6.08	6.12	5.87
Ours, r=.01	7.62	7.50	10.15	24.44	10.59	5.37
Ours, r=.001	10.29	11.23	18.34	50.50	23.98	5.26
Ours, r=0  	10.34	11.30	18.52	50.99	24.08	5.23
mean       	10.34	11.30	18.53	51.00	60.28	51.82

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	9.81	7.56	7.82	7.56	7.32	7.26
Ours, r=.01	11.14	9.18	12.22	212.44	28.73	13.42
Ours, r=.001	13.67	13.02	21.31	444.37	57.53	25.03
Ours, r=0  	13.72	13.12	21.50	149.63	58.12	25.10
mean       	13.72	13.12	21.50	149.64	96.87	114.39

Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	6.97	6.07	13.17	6.15	6.13	5.85
Ours, r=.01	7.92	7.55	112.45	54.57	15.02	7.16
Ours, r=.001	10.43	11.06	29.42	113.60	27.92	9.93
Ours, r=0  	10.49	11.15	29.59	64.76	29.41	9.92
mean       	10.49	11.15	29.59	64.77	57.05	58.75

Maxes:
N          	10^1	10^2	10^3	10^4	10^5	10^6
Ours, r=.1 	12.41	8.05	187.18	10.73	8.26	8.85
Ours, r=.01	13.75	10.26	2584.01	591.49	76.23	30.70
Ours, r=.001	16.17	15.15	301.62	1239.91	158.14	62.38
Ours, r=0  	16.22	15.26	301.58	250.56	160.00	63.65
mean       	16.22	15.26	301.58	250.58	160.01	176.32


Running: [1.e+00 3.e+00 1.e-04] [4.99975001e-01 4.99975001e-01 4.99975001e-05]
sqrt trace: 0.06324451215002738 6.002932102197005

Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.45	6.03	13.33	5.94	5.88	6.08
smoothedMLE-10	6.21	5.99	6.10	5.76	5.83	6.07
MLE        	9.91	11.30	30.02	62.58	35.75	10.71
MLE-10     	9.91	11.29	30.00	62.53	18.77	0.06
mean       	9.91	11.30	30.02	62.59	61.59	59.85

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	8.36	7.25	7.67	7.18	7.20	7.30
smoothedMLE-10	7.74	7.14	7.57	6.93	7.16	7.36
MLE        	12.22	13.67	21.77	147.15	126.30	29.09
MLE-10     	12.22	13.66	21.75	147.04	126.23	0.08
mean       	12.23	13.67	21.77	147.16	126.31	121.19

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.10	6.07	6.19	5.98	5.82	6.07
smoothedMLE-10	6.05	6.02	6.05	5.85	5.71	6.13
MLE        	9.70	11.29	18.76	51.00	24.45	5.06
MLE-10     	9.69	11.28	18.74	50.97	0.07	0.06
mean       	9.70	11.29	18.76	51.01	61.16	50.58

Maxes:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	12.20	8.48	189.69	8.09	8.30	8.70
smoothedMLE-10	10.50	8.51	8.53	7.62	8.19	8.36
MLE        	14.67	16.89	303.40	154.07	222.46	78.98
MLE-10     	14.66	16.88	303.40	153.95	222.32	0.09
mean       	14.67	16.89	303.40	154.08	222.48	197.43


Running: [1.   3.   0.01] [4.99975001e-01 4.99975001e-01 4.99975001e-05]
sqrt trace: 4.353966562510116 6.003209275047216
Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.53	15.57	18.65	6.24	5.99	5.98
smoothedMLE-10	6.18	14.11	5.94	6.14	5.94	5.96
MLE        	7.41	17.12	190.38	35.82	18.15	8.20
MLE-10     	6.15	15.77	552.89	81.78	4.35	4.34
mean       	10.10	20.75	38.36	56.05	59.25	64.89

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	8.68	7.09	8.12	7.74	7.17	7.25
smoothedMLE-10	7.96	7.00	7.27	7.51	7.14	7.29
MLE        	9.92	8.67	12.78	26.84	48.77	15.43
MLE-10     	7.84	6.99	7.97	7.43	5.33	5.38
mean       	13.08	13.36	21.93	54.63	127.95	123.33

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.25	5.99	6.14	6.17	6.04	6.03
smoothedMLE-10	5.96	5.89	5.97	6.24	6.04	5.96
MLE        	7.27	7.26	10.48	24.44	6.14	5.49
MLE-10     	5.87	5.89	6.07	5.09	4.20	4.33
mean       	9.92	10.93	19.19	51.03	35.36	57.68

Maxes:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	14.82	970.30	188.31	9.54	8.46	8.31
smoothedMLE-10	11.16	828.09	9.29	9.21	8.46	8.32
MLE        	14.95	993.16	2597.26	591.84	109.20	38.91
MLE-10     	11.56	996.03	7862.83	5379.80	7.14	6.37
mean       	15.11	992.64	302.54	250.52	194.74	200.10


Scaled errors [L2 error is C/sqrt(n) for the given C]
Reminder: [1. 3.] [0.5 0.5]
sqrt trace: 6.002600115761114 6.030304398466833

Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.55	5.86	6.06	5.85	6.13	5.89
smoothedMLE-adapt	6.76	5.85	6.02	5.82	6.11	5.86
smoothedMLE-10	6.15	5.82	6.05	5.84	6.14	5.89
MLE        	6.50	5.83	6.02	5.82	6.11	5.86
MLE-10     	6.14	5.79	6.01	5.82	6.11	5.86
mean       	9.75	9.84	9.85	9.94	10.20	9.96

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	8.49	6.94	7.38	7.10	7.91	7.10
smoothedMLE-adapt	9.09	7.03	7.33	7.07	7.75	6.90
smoothedMLE-10	7.99	6.88	7.27	7.05	7.77	7.03
MLE        	8.51	6.96	7.33	7.07	7.76	6.90
MLE-10     	8.11	6.68	7.30	7.08	7.76	6.90
mean       	12.30	11.74	12.01	11.85	12.21	12.20

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.22	5.90	5.99	5.88	6.10	5.97
smoothedMLE-adapt	6.42	5.85	6.01	5.82	6.14	6.03
smoothedMLE-10	5.92	5.89	5.99	5.79	6.14	6.00
MLE        	6.11	5.87	6.01	5.83	6.15	6.01
MLE-10     	5.92	5.81	6.01	5.83	6.15	6.01
mean       	9.56	9.60	9.91	9.95	10.30	9.97

Maxes:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	13.98	8.36	8.20	8.36	9.09	7.69
smoothedMLE-adapt	13.68	8.42	8.13	8.12	9.02	7.83
smoothedMLE-10	11.63	9.37	8.23	8.02	9.06	7.71
MLE        	13.77	8.26	8.12	8.10	9.02	7.85
MLE-10     	11.48	9.37	8.13	8.10	9.02	7.85
mean       	14.99	12.73	13.60	14.83	13.94	13.44



sqrt(Tr(I_R^{-1})) is 6.00
sqrt(Tr(I^{-1})) is   4.35

In the below tables we list the L2 error times sqrt(N).


d=20, MLE-10 is 10-step newton, small r is 1e-4

sqrt trace: 0.06324641582321536 6.002967665302243

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.02	6.09	6.55	7.00	8.21	19.52
smoothedMLE-10	5.55	5.80	5.90	6.00	5.76	5.84
MLE        	11.81	24.52	71.31	223.44	691.15	2180.29
MLE-10     	11.80	24.50	71.24	223.22	690.47	2178.13
mean       	11.81	24.52	71.32	223.46	691.23	2180.53

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	9.08	8.03	8.83	8.78	11.56	24.71
smoothedMLE-10	7.40	7.02	7.41	7.24	7.26	6.76
MLE        	15.16	27.02	75.27	262.31	699.22	2184.81
MLE-10     	15.16	27.00	75.20	262.08	698.54	2182.65
mean       	15.16	27.02	75.28	262.34	699.30	2185.06


Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.61	6.26	20.11	6.96	8.54	19.70
smoothedMLE-10	5.86	5.83	6.03	6.00	5.78	5.78
MLE        	12.04	24.51	90.87	228.36	692.69	2180.82
MLE-10     	12.03	24.49	90.80	228.14	692.01	2178.66
mean       	12.04	24.52	90.87	228.38	692.77	2181.06

Maxes:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	16.11	10.34	270.31	11.25	15.95	27.84
smoothedMLE-10	13.32	8.94	8.17	9.50	8.18	7.68
MLE        	18.41	31.55	617.47	328.96	723.55	2188.56
MLE-10     	18.41	31.52	617.47	328.69	722.85	2186.40
mean       	18.42	31.55	617.47	328.99	723.62	2188.80

d=20, MLE-10 is 10-step newton, small r is 1e-2

Means:
N          	10^1	10^2	10^3	10^4
smoothedMLE	6.46	6.04	13.25	5.96
smoothedMLE-10	6.23	5.99	5.93	5.86
MLE        	7.19	7.52	113.69	43.17
MLE-10     	6.17	5.97	320.18	142.55
mean       	9.75	11.29	30.00	58.92

90th percentile
N          	10^1	10^2	10^3	10^4
smoothedMLE	8.21	7.59	7.24	7.55
smoothedMLE-10	8.07	7.38	7.13	7.51
MLE        	9.53	9.44	12.22	26.72
MLE-10     	7.80	7.38	7.19	7.71
mean       	12.22	14.01	22.32	54.32

Medians:
N          	10^1	10^2	10^3	10^4
smoothedMLE	6.44	6.05	5.87	5.84
smoothedMLE-10	6.31	5.89	5.83	5.73
MLE        	7.12	7.29	10.19	24.27
MLE-10     	6.23	5.89	5.88	5.19
mean       	9.81	11.10	18.33	51.43

Maxes:
N          	10^1	10^2	10^3	10^4
smoothedMLE	10.07	8.89	190.48	11.04
smoothedMLE-10	9.39	8.76	8.65	8.54
MLE        	11.37	11.00	2609.60	584.77
MLE-10     	9.29	8.82	7901.72	5360.91
mean       	13.48	15.89	304.15	248.29

d=20, MLE is just 1-step newton [but I is better at least]

Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.55	6.05	17.16	6.07	5.95	5.99
MLE        	7.41	7.50	164.93	47.22	15.43	7.67
mean       	9.95	11.25	35.83	61.76	57.15	60.29

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.36	5.91	6.26	6.07	6.01	5.90
MLE        	7.26	7.46	10.77	24.59	6.54	5.34
mean       	9.76	11.33	19.34	51.13	36.49	51.04

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	8.49	7.34	8.00	7.19	7.00	7.22
MLE        	9.72	9.66	12.80	210.17	30.20	15.40
mean       	12.51	13.89	22.38	148.44	99.14	120.70




d=20, MLE was bad [poor computation of I]
Means:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.394	5.947	9.689	6.143	6.027	5.922
MLE        	7.143	7.382	56.469	50.844	19.728	10.095
mean       	9.775	10.882	23.877	66.972	62.052	55.344

90th percentile
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	8.711	7.535	7.854	7.699	7.248	7.273
MLE        	10.298	9.211	12.845	180.567	55.131	19.654
mean       	12.968	13.989	21.187	150.868	128.027	101.941

Medians:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.085	5.848	5.995	6.014	5.937	5.916
MLE        	6.669	7.359	10.761	26.717	9.043	7.694
mean       	9.421	10.629	18.179	51.476	36.018	48.370

Maxes:
N          	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	13.717	8.697	188.446	11.732	8.997	9.627
MLE        	14.936	11.773	2312.908	516.259	152.858	39.783
mean       	16.155	15.345	301.738	250.045	254.929	236.452



d=10, MLE was bad [poor computation of I]
Means:
N        	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	4.645	14.307	11.493	4.528	4.289	4.240
MLE        	5.309	15.533	108.684	45.517	17.606	9.028
mean       	7.078	18.459	28.329	61.555	61.757	66.982

90th percentile
N        	10^1	10^2	10^3	10^4	10^5	10^6
smoothedMLE	6.352	5.705	5.915	5.866	5.546	5.463
MLE        	7.306	7.559	11.725	195.969	49.023	21.087
mean       	9.229	11.603	20.939	147.780	126.643	129.957

Maxes:
N=10^k        	0	1	2	3	4	5	6
smoothedMLE	13.732	9.206	1012.615	185.894	25.895	6.553	6.570
MLE        	13.575	10.034	991.917	2517.099	554.937	81.557	62.615
mean       	13.652	11.835	991.428	301.686	253.989	161.341	237.977



Means:
N=10^k         	0	1	2	3	4
smoothedMLE	6.886	4.517	4.353	18.735	4.699
MLE        	6.646	5.050	5.407	236.326	62.908
mean       	6.600	6.923	8.458	40.189	65.047

90th percentile
N=10^k     	0	1	2	3	4
smoothedMLE	11.755	5.678	5.733	6.581	5.576
MLE        	11.387	6.674	6.938	12.860	242.018
mean       	11.100	9.054	10.643	22.978	148.831


Maxes:
N          	0	1	2	3	4
smoothedMLE	14.628	13.220	7.384	186.690	23.905
MLE        	14.171	14.045	9.486	2885.673	650.240
mean       	14.514	15.171	13.167	303.309	250.432


Mean error:
N             10  100  1000 10000 100000 1000000
smoothed MLE  6.2 
MLE           
mean          

1
smoothedMLE 6.496722174901852 3.4778577170254916 11.67343783047181
MLE 6.219223948628082 3.395306300051443 11.228025072239005
mean 6.1297036414782875 3.0982937040328933 10.941716560016907
10**1
smoothedMLE 5.031346575271807 0.7099890740090246 6.015349332281776
MLE 5.886404029168922 0.8576943740845332 7.294812920009593
mean 7.4673095464694175 1.161112590648257 9.983352329140887
10**2
smoothedMLE 3.866988384461341 1.0264749279860619 5.283039370830163
MLE 4.823666920372203 1.219326862230218 6.5720781306542895
mean 7.199001609048219 1.5237635761011104 9.934309884550462
10**3
smoothedMLE 3.913652871497672 1.1638217344080026 6.236809545380479
MLE 97.09475914516123 276.07368375165953 925.3077903973044
mean 17.38812044562841 27.51282409409492 99.7370212426126
10**4
smoothedMLE 3.4026983936630657 0.8268898788384207 5.239539708173069
MLE 21.243949633994838 24.39220713426601 70.17804749456573
mean 22.429155065390365 11.677355078920916 46.91697001835909
10**5
smoothedMLE 4.2402805303417965 0.9881484009582239 5.940855543065901
MLE 6.851344657367941 4.66663045018615 16.137520867770462
mean 21.915396913917572 9.788400457429473 39.72172842065738

10**6
smoothedMLE 3.731614293358998 0.8022029632850453 4.981254338344876
MLE 3.7106675290619884 1.3822449483941437 6.189758480512883
mean 17.195751968961286 9.343479317079666 36.779079892177734
'''

'''
10**0	{'smoothedMLE': [1.9440166969235566, 2.541603827406068, 2.6778119968561156, 4.570721516130035, 4.773228465027208, 7.851000600571264, 8.254464502474672, 9.328047501178345, 11.352888811979446, 11.67343783047181], 'MLE': [1.7233346344324905, 2.3244621861309027, 2.513722638240057, 4.332068711162176, 4.752651119455028, 7.551644423457139, 7.6373069977737655, 9.062250055087233, 11.06677364830302, 11.228025072239005], 'mean': [1.981588377104923, 2.7737438747526633, 2.7980320483429284, 4.157438969056376, 4.84536154031125, 7.1619464457836735, 7.59423066461806, 8.652217417278983, 10.390760517517108, 10.941716560016907]}
smoothedMLE 6.496722174901852 3.4778577170254916 11.67343783047181
MLE 6.219223948628082 3.395306300051443 11.228025072239005
mean 6.1297036414782875 3.0982937040328933 10.941716560016907
10**1	{'smoothedMLE': [3.9302917199607994, 4.042641410114391, 4.351518941242174, 4.530438937357756, 5.316911780067899, 5.354577424027829, 5.423780929779073, 5.631141344584811, 5.716813933301565, 6.015349332281776], 'MLE': [4.456633025298384, 4.49077896517559, 5.270554171394906, 5.9313334048318636, 6.036726067770443, 6.090932796391504, 6.255478724668313, 6.516614618033755, 6.5201755981148715, 7.294812920009593], 'mean': [5.844054604785013, 6.015075448747201, 6.522312611947074, 7.234080384934217, 7.24182411959773, 7.385378785825717, 8.102928576845944, 8.121410121834765, 8.22267848103563, 9.983352329140887]}
smoothedMLE 5.031346575271807 0.7099890740090246 6.015349332281776
MLE 5.886404029168922 0.8576943740845332 7.294812920009593
mean 7.4673095464694175 1.161112590648257 9.983352329140887
10**2	{'smoothedMLE': [2.0998083503604104, 2.480640286649147, 3.157019465406119, 3.4413896397830657, 3.798346171677056, 4.269299876762264, 4.33309265245566, 4.537437886562416, 5.269810144127107, 5.283039370830163], 'MLE': [3.021078049115, 3.4363225759052147, 3.4721699097890495, 3.8620032094308456, 5.05399435443982, 5.066686974847979, 5.567353666580177, 6.081614756407415, 6.103367576552237, 6.5720781306542895], 'mean': [4.459788712750472, 5.835182634934686, 6.122875101916127, 6.604925990228638, 6.626616445339599, 7.495830067768229, 7.834134787345821, 8.060433623658573, 9.015918841989585, 9.934309884550462]}
smoothedMLE 3.866988384461341 1.0264749279860619 5.283039370830163
MLE 4.823666920372203 1.219326862230218 6.5720781306542895
mean 7.199001609048219 1.5237635761011104 9.934309884550462
10**3	{'smoothedMLE': [2.702126542296987, 2.785233501567007, 2.9620929714376634, 3.1069510046747504, 3.2278758527499414, 3.8047525549255203, 4.157049452006548, 4.527686433960508, 5.625950855977308, 6.236809545380479], 'MLE': [3.2441828295260824, 3.8182748174198813, 3.8282852502164824, 4.592423541932446, 4.920971262028695, 5.210824213105051, 6.224266306766294, 6.81068586227934, 6.989886971033672, 925.3077903973044], 'mean': [6.27088747517423, 6.461644065441307, 6.761917407592356, 6.796904808974049, 7.239459011570298, 8.686723251402862, 8.878109306930721, 11.17555797251872, 11.872979914066935, 99.7370212426126]}
smoothedMLE 3.913652871497672 1.1638217344080026 6.236809545380479
MLE 97.09475914516123 276.07368375165953 925.3077903973044
mean 17.38812044562841 27.51282409409492 99.7370212426126
10**4	{'smoothedMLE': [2.460200102744463, 2.4902758278916965, 2.6906722586181946, 3.0360069836277948, 3.0848111750993548, 3.206602625880296, 3.7035024339487883, 3.976404739538455, 4.138968081108544, 5.239539708173069], 'MLE': [6.6110408573453405, 7.516355134064559, 8.799520941569195, 9.002506384477185, 9.498305988574568, 9.609360294451088, 9.796196748364206, 11.684620031950358, 69.74354246458613, 70.17804749456573], 'mean': [13.939352125582019, 14.50263335950848, 15.271444263215692, 15.962567676711318, 18.11996881411311, 18.28924310217362, 18.54505577634067, 18.619740739518274, 44.12457477838138, 46.91697001835909]}
smoothedMLE 3.4026983936630657 0.8268898788384207 5.239539708173069
MLE 21.243949633994838 24.39220713426601 70.17804749456573
mean 22.429155065390365 11.677355078920916 46.91697001835909
10**5	{'smoothedMLE': [2.5956476131188726, 3.1817437684127614, 3.4055151448829313, 3.7254395230469726, 4.193659843857453, 4.329609952040828, 4.773350885715276, 4.802109464027254, 5.4548735652497164, 5.940855543065901], 'MLE': [3.618202646399785, 3.643202053297397, 3.8645383098038644, 4.3413359476960816, 4.7856732071378305, 4.9004872768315115, 5.524577411150325, 5.663404082743705, 16.034504770848436, 16.137520867770462], 'mean': [10.21188478721809, 11.3313685622876, 12.317237908484575, 19.441297712657335, 20.302448263362482, 22.134707692152833, 22.271137800866622, 22.45521988513648, 38.966938106352345, 39.72172842065738]}
smoothedMLE 4.2402805303417965 0.9881484009582239 5.940855543065901
MLE 6.851344657367941 4.66663045018615 16.137520867770462
mean 21.915396913917572 9.788400457429473 39.72172842065738

10**6	{'smoothedMLE': [2.3020600339024315, 2.882658940726527, 3.1642158624217784, 3.5000614648149053, 3.524040057857284, 3.593626376817106, 4.31331085563104, 4.34503492241196, 4.7098800806620735, 4.981254338344876], 'MLE': [1.5582745562430467, 2.1775964367217915, 3.0617638870002972, 3.331271266282925, 3.359668334499975, 3.47637793410248, 3.5500051722982615, 4.5820957630159524, 5.819863459942271, 6.189758480512883], 'mean': [6.422522055174354, 6.999112978832075, 8.269563763799642, 9.167911320231724, 15.348595280624254, 20.16706515044391, 20.48151013756888, 22.537216761841247, 25.784942348919056, 36.779079892177734]}
smoothedMLE 3.731614293358998 0.8022029632850453 4.981254338344876
MLE 3.7106675290619884 1.3822449483941437 6.189758480512883
mean 17.195751968961286 9.343479317079666 36.779079892177734
'''
