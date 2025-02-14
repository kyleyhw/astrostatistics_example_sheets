import scipy
import numpy as np
import matplotlib.pyplot as plt

def gamma(y):
    result = 1/np.log(y) + 1
    print(result)
    return result

gamma = np.vectorize(gamma)

data = np.loadtxt('ex1_data_for_problem2.txt')
data = np.array([data])
print(data)
res = scipy.stats.bootstrap(data=data, statistic=gamma, n_resamples=2)

print(res.bootstrap_distribution)

plt.figure()

plt.hist(res.bootstrap_distribution, bins=40)

plt.show()