import numpy as np

def generate_parameters(lam, mu, lam_tilt, mu_tilt):
    """
    return a 2d-array of the input parameters
    """
    return np.array(list(zip(lam, lam_tilt))), np.array(list(zip(mu, mu_tilt))), len(lam)

def mean_CI(count):
    mean = np.mean(count)
    halfCI = 1.96 * np.std(count) / np.sqrt(len(count))
    return mean, halfCI

# if numerator and denominator are independent
def cycle_sta1(cycle_num_mean, cycle_num_var, cycle_sum):
    N = len(cycle_sum)
    R_hat = np.mean(cycle_sum)
    l_hat = R_hat / cycle_num_mean
    S11 = np.sum((cycle_sum - R_hat)**2) / (N-1)
    S22 = cycle_num_var
    S_2 = S11 + l_hat**2 * S22 
    half_CI = 1.96 * np.sqrt(S_2 / N) / cycle_num_mean
    return l_hat, half_CI

# if numerator and denominator are dependent
def cycle_statistics(cycle_num, cycle_sum):
    N = len(cycle_num)
    R_hat = np.mean(cycle_sum)
    tau_hat = np.mean(cycle_num)
    l_hat = R_hat / tau_hat
    S11 = np.sum((cycle_sum - R_hat)**2) / (N-1)
    S22 = np.sum((cycle_num - tau_hat)**2) / (N-1)
    S12 = np.sum((cycle_num - tau_hat)*(cycle_sum - R_hat)) / (N-1)
    S_2 = S11 - 2*l_hat*S12 + l_hat**2 * S22 
    half_CI = 1.96 * np.sqrt(S_2 / N) / tau_hat
    return l_hat, half_CI

def combine_result(results:dict, result_batch:list[dict]):
    for result in result_batch:
        for k,v in result.items():
            results[k].append(v)