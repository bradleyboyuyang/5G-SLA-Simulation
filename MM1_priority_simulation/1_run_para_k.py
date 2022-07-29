import argparse
import datetime
import time
import numpy as np
import ray
import csv 
from multiprocessing import Pool, freeze_support, cpu_count

import warnings
warnings.filterwarnings('ignore')
from distribution import ExponDistri
from simulation import static_priority_sim

def generate_parameters(lam, mu, lam_tilt, mu_tilt):
    """
    return a 2d-array of the input parameters
    """
    return np.array(list(zip(lam, lam_tilt))), np.array(list(zip(mu, mu_tilt))), len(lam)

def mean_CI(count):
    mean = np.mean(count)
    halfCI = 1.96 * np.std(count) / np.sqrt(len(count))
    return mean, halfCI

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

def get_flow_result(num_type, start_local, flow_id:int, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L:list, start):
    """
    target: obtain typical statistics of the events with type "flow_id"
    theory: theoretical waiting time
    """
    L[flow_id].extend([args.n_cycles, time.time()-start_local, sum(nums)])
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]

    # statistics of sojourn time
    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow {flow_id+1} (Denominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id+1} (Numerator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id+1}: {mean:.14g}s  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {theory:.14g}s)')
    print("=====================================================================================================================================")

    # statistics of probability
    for i, gamma in enumerate(threshold[flow_id]):
        cycle_sum_prob = np.array([cycle_sum_prob[flow_id][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        L[flow_id].extend([gamma, mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob (gamma={gamma}) in flow {flow_id+1} (Numerator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob (gamma={gamma}) in flow {flow_id+1}: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[flow_id][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)

        # supplementary statistics
        for k in range(num_type):
            print(f'Mean flow {k+1}: {np.mean(cycle_sum_prob[freq==k])}')
            print(f'Std flow {k+1}: {np.std(cycle_sum_prob[freq==k])}')
            print(f'ratio flow {k+1}: {sum(freq==k)/len(freq)}')
            L[flow_id].extend([sum(freq==k)/len(freq)]) # mean flow (ratio) of type k
            L[flow_id].extend([np.mean(cycle_sum_prob[freq==k]), np.var(cycle_sum_prob[freq==k])]) # likelihood, variance of type k event
            L[flow_id].extend([sum(freq==k)/(sum(freq!=-1))]) # conditional ratio of type k event
        L[flow_id].extend([sum(freq!=-1)/len(freq), np.var(cycle_sum_prob)]) # total ratio and total variance
        print("=====================================================================================================================================")
    return L

def main(lam, mu, port_rate, threshold, args, mix_p, num_type, start):
    start_local = time.time()
    L = [[] for i in range(num_type)]
    L[0].append(args.idx)
    if mix_p is not None:
        L[0].extend(list(mix_p.keys()) + list(mix_p.values()))
    for i in range(num_type):
        L[i].extend(list(lam[i])+list(mu[i]))
    
    lam_each = lam[:,0]
    mu_each = mu[:,0]
    p = lam / np.sum(lam, axis=0, keepdims=True)
    p_each = p[:,0]
    rho_each = lam_each / mu_each
    rho = np.sum(rho_each)
    ES, ES2 = np.sum(p_each/mu_each), 2*np.sum(p_each/mu_each**2)
    Se = ES2 / ES / 2

    # simulation starts
    print("=====================================================================================================================================")
    print(f'({time.time()-start_local:.2f}s)Run simulation...')
    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, args.n_cycles, start, p=mix_p, policy=args.policy)
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums = result[:5]

    # obtain statistics
    print("=====================================================================================================================================")
    print(f'({time.time()-start_local:.2f}s)Statistics:')
    for flow_id in range(num_type):
        # calculate theoretical sojourn time (mean waiting time + service time)
        theory = rho*Se/(1-rho_each[:flow_id+1].sum())/(1-rho_each[:flow_id].sum()) + 1/mu_each[flow_id]
        # obtain simulation result for each type of events
        L = get_flow_result(num_type, start_local, flow_id, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L, start)
    print("total time: ", time.time()-start)
    print("summation of nums: ", sum(nums))
    print("=====================================================================================================================================")

    # append the simulation record into a csv file
    with open("result/1_run_para_1.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows([sum(L, [])])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=cpu_count(), help='the number of cpus')
    parser.add_argument('--test_flow', type=int, default=0, help='which flow to switch measure')
    parser.add_argument('--idx', type=int, default=None, help='index of the instance')
    parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
    args = parser.parse_args()

    # # 2-class example:
    # lam = [0.6, 0.2]
    # mu = [2, 1]
    # lam_tilt = [0.8333, 0.4666]
    # mu_tilt = [1.2, 0.2]

    # # thresholds (must be increasing for each flow)
    # threshold = [[5], [5, 10]]

    # 4-class example:
    lam = [0.6, 0.2, 0.2, 0.2]
    mu = [2, 1, 2, 2]
    lam_tilt = [0.804031094895019, 0.532834011679204, 0.267738894790881, 0.293051762204003]
    mu_tilt = [1.3875321736264, 0.498272105811342, 1.11181721710237, 1.01934603044239]
    # lam_tilt = [0.757056038322328, 0.320870841120882, 0.238073067036217, 0.359192084875008]
    # mu_tilt = [1.45755526837829, 0.620977255373011, 1.49215499325418, 1.23569266294702]
    # thresholds (must be increasing for each flow)
    threshold = [[2, 3], [2, 3], [2, 3], [2, 3]]

    # initialization
    np.random.seed(42)
    mix_p = None
    lam, mu, num_type = generate_parameters(lam=lam, mu=mu, lam_tilt=lam_tilt, mu_tilt=mu_tilt)
    start = time.time()
    port_rate = 1 * 8 # 1 byte per second
    if args.cpu_nums is not None and args.cpu_nums > 1:
        ray.init(num_cpus=args.cpu_nums)
    if args.idx is None:
        args.idx = f'{datetime.datetime.now()}'
    main(lam, mu, port_rate, threshold, args, mix_p, num_type, start) 

