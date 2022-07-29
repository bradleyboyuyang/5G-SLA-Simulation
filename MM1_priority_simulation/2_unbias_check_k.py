import argparse
import csv
import datetime
import time
import numpy as np
import ray
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

def get_flow_result(num_type, start_local, flow_id:int, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L:list, start, unbias_check, real, real_lower, real_upper):
    """
    target: obtain typical statistics of the events with type "flow_id"
    theory: theoretical waiting time
    """
    L[flow_id].extend([args.n_cycles, time.time()-start_local, sum(nums)])
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]

    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow {flow_id+1} (Denominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id+1} (Numerator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id+1}: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {theory:.14g} s)')
    print("=====================================================================================================================================")

    for i, gamma in enumerate(threshold[flow_id]):
        cycle_sum_prob = np.array([cycle_sum_prob[flow_id][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)

        # 1.difference is smaller than half CI; 2. two CIs should overlap
        unbias_check.extend([abs(mean-real[flow_id][i])<halfCI, mean-halfCI<real_upper[flow_id][i] and mean+halfCI>real_lower[flow_id][i]])
        L[flow_id].extend([gamma, mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob (gamma={gamma}) in flow {flow_id+1} (Numerator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob (gamma={gamma}) in flow {flow_id+1}: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[flow_id][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)

        for k in range(num_type):
            print(f'Mean flow {k+1}: {np.mean(cycle_sum_prob[freq==k])}')
            print(f'Std flow {k+1}: {np.std(cycle_sum_prob[freq==k])}')
            print(f'ratio flow {k+1}: {sum(freq==k)/len(freq)}')
            L[flow_id].extend([sum(freq==k)/len(freq)]) # mean flow (ratio) of type k
            L[flow_id].extend([np.mean(cycle_sum_prob[freq==k]), np.var(cycle_sum_prob[freq==k])]) # likelihood, variance of type k event
            L[flow_id].extend([sum(freq==k)/(sum(freq!=-1))]) # conditional ratio of type k event
        L[flow_id].extend([sum(freq!=-1)/len(freq), np.var(cycle_sum_prob)]) # total ratio and total variance
        print("=====================================================================================================================================")
    return L, unbias_check

def main(idx, lam, mu, port_rate, threshold, args, mix_p, num_type, start, real, real_lower, real_upper):
    unbias_check = []
    start_local = time.time()
    L = [[] for i in range(num_type)]
    L[0].append(idx)
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
    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, args.n_cycles, start_local, p=mix_p, policy=args.policy)
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums = result[:5]

    # obtain statistics
    print("=====================================================================================================================================")
    print(f'({time.time()-start_local:.2f}s)Statistics:')
    for flow_id in range(num_type):
        # calculate theoretical sojourn time (mean waiting time + service time)
        theory = rho*Se/(1-rho_each[:flow_id+1].sum())/(1-rho_each[:flow_id].sum()) + 1/mu_each[flow_id]
        # obtain simulation result for each type of events
        L, unbias_check = get_flow_result(num_type, start_local, flow_id, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L, start, unbias_check, real, real_lower, real_upper)
    print("total time: ", time.time()-start)
    print("summation of nums: ", sum(nums))
    print("unbias_check: ", unbias_check)
    print("=====================================================================================================================================")

    with open(f"result/2_unbias_check/{args.file_name}_t{args.test_flow}_iter{args.num_runs}_policy{args.policy}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows([sum(L, [])])
    print(f'({time.time()-start_local:.2f}s) END')
    return unbias_check

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
    parser.add_argument('-t', '--test_flow', type=int, default=1, help='which flow to switch measure')
    parser.add_argument('-k', '--num_runs', type=int, default=10, help='number of runs')
    parser.add_argument('-f', '--file_name', type=str, default='temp')
    parser.add_argument('--policy', type=int, default=1, help='policy of switching measure')
    args = parser.parse_args()

    # # 2-class example:
    # lam = [0.6, 0.2]
    # mu = [2, 1]
    # lam_tilt = [None, None]
    # mu_tilt = [None, None]
    # lam, mu, num_type = generate_parameters(lam=lam, mu=mu, lam_tilt=lam_tilt, mu_tilt=mu_tilt)

    # parameters
    # ### Priority 5
    # lam[:,1] = [0.8998870926555549,0.6142637224394023]
    # mu[:,1] = [1.0758906952016303,0.2424444477171096]

    # # Ordinary 10 policy 0
    # lam[:,1] = [0.8204897498205567,0.5791510463854309]
    # mu[:,1] = [1.3366737345090036,0.4373962955149862]

    # # Ordinary 10 policy 1
    # lam[:,1] = [0.8017380259860238,0.579801196549211]
    # mu[:,1] = [1.3665636797944918,0.4237711449316795]

    # ## Ordinary 10 policy 2
    # lam[:,1] = [0.8394579259348436,0.5982263254131608]
    # mu[:,1] = [1.3421835386475596,0.4413138790146214]

    threshold = [[5], [5, 10]]
    real = [[0.0050488825], [0.025138833333333, 0.0038560725]]
    real_lower = [[0.0050349732904737], [0.024898493401137,0.0038443439681686]]
    real_upper = [[0.0050627917095263], [0.025379173265529,0.0038678010318314]]

    # 4-class example:
    lam = [0.6, 0.2, 0.2, 0.2]
    lam_tilt = [None, None, None, None]
    mu = [2, 1, 2, 2]
    mu_tilt = [None, None, None, None]
    lam, mu, num_type = generate_parameters(lam=lam, mu=mu, lam_tilt=lam_tilt, mu_tilt=mu_tilt)
    lam[:,1] = [0.757056038322328, 0.320870841120882, 0.238073067036217, 0.359192084875008]
    mu[:,1] = [1.45755526837829, 0.620977255373011, 1.49215499325418, 1.23569266294702]

    ## thresholds (must be increasing for each flow)
    threshold = [[2, 3], [2, 3], [2, 3], [2, 3]]
    real = [[0.1381516101, 0.04882982104], [0.12821282098, 0.08244167776], [0.18036646152, 0.13630755494], [0.21625822012, 0.17811018112]]
    real_lower = [[0.135866, 0.047715], [0.12613, 0.0809764], [0.177152, 0.133726], [0.21243, 0.174728]]
    real_upper = [[0.14044, 0.0499446], [0.130302, 0.0839068], [0.183584, 0.138894], [0.220088, 0.181498]]

    # initialization
    np.random.seed(42)
    mix_p = None
    start = time.time()
    port_rate = 1 * 8 # 1 byte per second
    if args.cpu_nums is not None and args.cpu_nums > 1:
        ray.init(num_cpus=args.cpu_nums)

    unbias_checks = [main(idx, lam, mu, port_rate, threshold, args, mix_p, num_type, start, real, real_lower, real_upper) for idx in range(args.num_runs)]
    res = np.mean(unbias_checks, axis=0)
    print(res)
    with open(f"result/2_unbias_check/{args.file_name}_t{args.test_flow}_iter{args.num_runs}_policy{args.policy}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([res])
