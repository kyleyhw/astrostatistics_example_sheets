import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def gamma(ys):
    N = len(ys)
    result = 1 + N / np.sum(ys)
    return result

def calculate_statistics(bootstrap_res):
    mean = np.mean(bootstrap_res.bootstrap_distribution)
    confidence_interval = (bootstrap_res.confidence_interval[0], bootstrap_res.confidence_interval[1])
    standard_error = bootstrap_res.standard_error
    cramer_rao_lower_bound = (mean - 1)**2 / len(data)
    return mean, confidence_interval, standard_error, cramer_rao_lower_bound

def plot_histogram(mean, confidence_interval, standard_error, cramer_rao_lower_bound,
                   save=True, show=False):
    info_on_ax = 'mean = ' + str(mean) + \
                 '\nconfidence interval = ' + str(confidence_interval) + \
                 '\nstandard error = ' + str(standard_error) + \
                 '\nCRLB = ' + str(cramer_rao_lower_bound)

    info_fontsize = 14
    info_loc = 'upper left'

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.hist(bootstrap_res.bootstrap_distribution, bins=40)

    ax_text = AnchoredText(info_on_ax, loc=info_loc, frameon=False, prop=dict(fontsize=info_fontsize))
    ax.add_artist(ax_text)

    ax.axvline(mean, c='orange')
    ax.axvline(confidence_interval[0], c='black')
    ax.axvline(confidence_interval[1], c='black')

    ax.set_ylabel('count')
    ax.set_xlabel(r'$\hat{\gamma}$')

    plt.savefig('q2_bootstrap_histogram')

    plt.show()


np.random.seed(seed=1)

data = np.loadtxt('ex1_data_for_problem2.txt')
bootstrap_res = scipy.stats.bootstrap(data=(data,), statistic=gamma, n_resamples=9999)

statistics = calculate_statistics(bootstrap_res=bootstrap_res)
plot_histogram(*statistics, save=True, show=True)