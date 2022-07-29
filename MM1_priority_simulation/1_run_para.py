
import argparse
import datetime
import time

import numpy as np
import ray

from distribution import ExponDistri
from simulation import static_priority_sim

parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of the training dataset')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('--test_flow', type=int, default=0, help='which flow to swtich measure')
parser.add_argument('--idx', type=int, default=None, help='index of the instance')
parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
args = parser.parse_args()


# parameters
mix_p = None
num_type = 2
## lam mu   original       1
lam = np.array([[0.6, 0.8333],
                [0.2, 0.4666]])
mu = np.array([[   2,    1.2],
               [   1,    0.2]])
## thresholds (must be increasing for each flow)
threshold = [[5], [5, 10]]

# initial
np.random.seed(42)
start = time.time()
port_rate = 1 * 8 # 1 byte per second
if args.cpu_nums is not None and args.cpu_nums > 1:
    ray.init(num_cpus=args.cpu_nums)
if args.idx is None:
    args.idx = f'{datetime.datetime.now()}'

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

def main(lam, mu, port_rate, threshold, args, mix_p, num_type):
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
    rho_total = np.sum(rho_each)
    lam1, lam2 = lam_each
    mu1, mu2 = mu_each
    rho = rho_total
    rho1, rho2 = rho_each

    print(f'({time.time()-start_local:.2f}s)Run simulation...')
    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, args.n_cycles, start, p=mix_p, policy=args.policy)
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums = result[:5]

    print(f'({time.time()-start_local:.2f}s)Statistics:')

    flow_id = 0
    L[flow_id].extend([args.n_cycles, time.time()-start_local, sum(nums)])
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]
            
    ES, ES2 = np.sum(p_each/mu_each), 2*np.sum(p_each/mu_each**2)
    Se = ES2 / ES / 2
    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow 1 (Denominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow 1 (Nominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow 1: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {rho*Se/(1-rho1)+1/mu1:.14g} s)')

    for i, gamma in enumerate(threshold[flow_id]):
        cycle_sum_prob = np.array([cycle_sum_prob[flow_id][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        L[flow_id].extend([gamma, mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob {gamma} in flow 1 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob {gamma} in flow 1: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[0][i] for reason in reasons])
        print(f'Mean flow 1 {np.mean(cycle_sum_prob[freq==0])}')
        print(f'ratio flow 1 {sum(freq==0)/len(freq)}')
        print(f'Mean flow 2 {np.mean(cycle_sum_prob[freq==1])}')
        print(f'ratio flow 2 {sum(freq==1)/len(freq)}')
        L[flow_id].extend([sum(freq==0)/len(freq), sum(freq==1)/len(freq), sum(freq==0)/len(freq) + sum(freq==1)/len(freq), sum(freq==1)/(sum(freq==0)+sum(freq==1))])
        L[flow_id].extend([np.mean(cycle_sum_prob[freq==0]), np.mean(cycle_sum_prob[freq==1])])
        L[flow_id].extend([np.var(cycle_sum_prob[freq==0]), np.var(cycle_sum_prob[freq==1])])
        L[flow_id].append(np.var(cycle_sum_prob))
        print(f'Frquency reason {gamma} in flow 1 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')


    flow_id = 1
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]
    # cycle_sum_prob = packets_info[:, 5]
    # reasons = [info[3][1] for info in packets_info]    
    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow 2 (Denominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow 2 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow 2: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {rho*Se/(1-rho1)/(1-rho)+1/mu2:.14g} s) ')
    for i, gamma in enumerate(threshold[1]):
        cycle_sum_prob = np.array([cycle_sum_prob[1][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob {gamma} in flow 2 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        print(f'std {np.std(cycle_sum_prob)}')
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob {gamma} in flow 2: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[1][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)
        print(f'Mean flow 1 {np.mean(cycle_sum_prob[freq==0])}')
        print(f'Std flow 1 {np.std(cycle_sum_prob[freq==0])}')
        print(f'ratio flow 1 {sum(freq==0)/len(freq)}')
        print(f'Mean flow 2 {np.mean(cycle_sum_prob[freq==1])}')
        print(f'Std flow 2 {np.std(cycle_sum_prob[freq==1])}')
        print(f'ratio flow 2 {sum(freq==1)/len(freq)}')
        L[flow_id].extend([sum(freq==0)/len(freq), sum(freq==1)/len(freq), sum(freq==0)/len(freq) + sum(freq==1)/len(freq), sum(freq==1)/(sum(freq==0)+sum(freq==1))])
        L[flow_id].extend([np.mean(cycle_sum_prob[freq==0]), np.mean(cycle_sum_prob[freq==1])])
        L[flow_id].extend([np.var(cycle_sum_prob[freq==0]), np.var(cycle_sum_prob[freq==1])])
        L[flow_id].append(np.var(cycle_sum_prob))
        print(f'Frquency reason {gamma} in flow 2 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    print(time.time()-start)
    print(sum(nums))
    

    # # print(f'Mean interarrival time in flow 1: {pg1.time_rec[-1] / len(pg1.time_rec):.4f} s (theoretic {1/lam1:.4f} s)')
    # # print(f'Mean interarrival time in flow 2: {pg2.time_rec[-1] / len(pg2.time_rec):.4f} s (theoretic {1/lam2:.4f} s)')
    # # print(f'Mean packet length in flow 1: {np.mean(pg1.size_rec):.4f} bytes (theoretic {1/mu1:.4f}) bytes')
    # # print(f'Mean packet length in flow 2: {np.mean(pg2.size_rec):.4f} bytes (theoretic {1/mu2:.4f}) bytes')
    # # ES, ES2 = p1/mu1 + p2/mu2, 2*p1/mu1**2 + 2*p2/mu2**2 
    # # Se = ES2 / ES / 2
    # # waits = [packet.response for packet in ps.packets[0]]
    # # print(f'Mean response time in flow 1: {np.mean(waits):.4f} s (theoretic {rho*Se/(1-rho1)+1/mu1:.4f} s)')
    # # waits = [(packet.wait > 5) for packet in ps.packets[0]]
    # # print(f'Prob in flow 1: {np.mean(waits):.4f}')
    # # waits = [packet.response for packet in ps.packets[1]]
    # # print(f'Mean response time in flow 2: {np.mean(waits):.4f} s (theoretic {rho*Se/(1-rho1)/(1-rho)+1/mu2:.4f} s)')

    import csv 
    with open("result/1_run_para_1.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerows([sum(L, [])])

if __name__ == '__main__':
    main(lam, mu, port_rate, threshold, args, mix_p, num_type) 

