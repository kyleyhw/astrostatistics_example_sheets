import numpy as np
import scipy
from bootstrap_funcs import calculate_statistics, plot_histogram, calculate_cramer_rao_lower_bound

def gamma(ys):
    N = len(ys)
    result = 1 + N / np.sum(np.log(ys))
    return result



np.random.seed(seed=1)

data = np.loadtxt('sheet_1_folder_download\ex1_data_for_problem2.txt')
bootstrap_res = scipy.stats.bootstrap(data=(data,), statistic=gamma, n_resamples=9999)

statistics = calculate_statistics(bootstrap_res=bootstrap_res)
mean, confidence_interval, standard_error = statistics
cramer_rao_lower_bound = calculate_cramer_rao_lower_bound(mle=gamma(data), data=data)

plot_histogram(bootstrap_res=bootstrap_res,
               mean=mean, confidence_interval=confidence_interval, standard_error=standard_error,
               cramer_rao_lower_bound=cramer_rao_lower_bound,
               save=True, show=False)