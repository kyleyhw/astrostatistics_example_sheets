import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def gamma(ys):
    N = len(ys)
    result = 1 + N / np.sum(ys)
    return result

data = np.loadtxt('ex1_data_for_problem2.txt')
bootstrap_res = scipy.stats.bootstrap(data=(data,), statistic=gamma, n_resamples=9999)

mean = np.mean(bootstrap_res.bootstrap_distribution)
confidence_interval = (bootstrap_res.confidence_interval[0], bootstrap_res.confidence_interval[1])
standard_error = bootstrap_res.standard_error

info_on_ax = 'mean = ' + str() + \
             '\nconfidence interval = ' + str(confidence_interval) + \
             '\nstandard error = ' + str(standard_error)
info_fontsize = 14
info_loc = 'upper left'

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

ax.hist(bootstrap_res.bootstrap_distribution, bins=40)

ax_text = AnchoredText(info_on_ax, loc=info_loc, frameon=False, prop=dict(fontsize=info_fontsize))
ax.add_artist(ax_text)

ax.axvline(mean, c='orange')
ax.axvline(confidence_interval[0], c='black')
ax.axvline(confidence_interval[1], c='black')

plt.savefig('q2_bootstrap_histogram')

plt.show()