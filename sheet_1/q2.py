import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# def gamma(ys):
#     result = np.prod(1/np.log(ys) + 1)
#     print(result)
#     return result

def gamma(ys):
    N = len(ys)
    result = 1 + N / np.sum(ys)
    return result

data = np.loadtxt('ex1_data_for_problem2.txt')
bootstrap_res = scipy.stats.bootstrap(data=(data,), statistic=gamma, n_resamples=9999)

info_on_ax = 'standard error = ' + str(bootstrap_res.standard_error)
info_fontsize = 14
info_loc = 'upper left'

fig, ax = plt.subplots()

ax.hist(bootstrap_res.bootstrap_distribution, bins=40)

ax_text = AnchoredText(info_on_ax, loc=info_loc, frameon=False, prop=dict(fontsize=info_fontsize))
ax.add_artist(ax_text)

plt.show()