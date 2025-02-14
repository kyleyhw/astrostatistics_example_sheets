import numpy as np
import scipy

class LogLikelihood:
    def __init__(self, distribution):
        self.distribution = distribution

    def __call__(self, data, params):
        class my_pdf(scipy.stats.rv_continuous):

        likelihood = np.prod([self.distribution(data_point, params) for data_point in data])
        log_likelihood = likelihood # np.log(likelihood)
        return log_likelihood

class MLE:
    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood

    def __call__(self, data):
        res = scipy.optimize.minimize(
            fun=lambda params, data: -self.log_likelihood(data, params),
            x0=np.array((5, 5)),
            args=(data,),
            method='BFGS'
        )
        params = res.x
        return params

class FittedDistribution:
    def __init__(self, distribution, fitted_params):
        self.distribution = distribution
        self.fitted_params = fitted_params

    def __call__(self, x):
        return self.distribution(x, self.fitted_params)


def truncated_pareto(x, params):
    b, c = params
    return scipy.stats.truncpareto.pdf(x, b, c)

def gaussian(x, params):
    mu, sigma = params
    return np.exp(-(x - mu)**2 / (2 * sigma**2))


np.random.seed(seed=1)
xs = np.linspace(-10, 10, 1000)

dist = gaussian
ys = dist(x=xs, params=(1, 5.0))

log_likelihood = LogLikelihood(truncated_pareto)
mle = MLE(log_likelihood=log_likelihood)
fitted_parameters = mle(ys)
print(fitted_parameters)

fitted_distribution = FittedDistribution(distribution=dist, fitted_params=fitted_parameters)
fit_ys = fitted_distribution(xs)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(xs, ys)
plt.plot(xs, fit_ys)

plt.show()