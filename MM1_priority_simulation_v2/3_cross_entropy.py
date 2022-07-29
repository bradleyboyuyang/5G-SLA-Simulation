import argparse
import time

import numpy as np
import pandas as pd

from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_statistics, mean_CI


parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
parser.add_argument('-n', '--n_cycles', type=int, default=10000, help='the size of the training dataset')
parser.add_argument('-N','--test_cycles', type=int, default=100000, help='the size of the training dataset')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('-t','--test_flow', type=int, default=0, help='the number of cpus')
parser.add_argument('-T','--threshold', type=float, default=10, help='threshold')
parser.add_argument('-k', '--num_runs', type=int, default=5, help='number of runs')
parser.add_argument('--rho', type=float, default=0.1, help='para of CE')
parser.add_argument('-f','--file_name', type=str, default='temp')
parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
args = parser.parse_args()

# initial

# parameters
mix_p = None
num_type = 2
## lam mu 
lam_ref = np.array([[0.6],[0.2]], dtype=float)
mu_ref = np.array([[2],[1]], dtype=float)
## thresholds (must be increasing for each flow)
threshold = [[0], [0]]
threshold[args.test_flow].append(args.threshold)

# initial
np.random.seed(42)
start = time.time()
port_rate = 1 * 8 # 1 byte per second



def main(lam, mu, port_rate, threshold, args, CoE_p):
    start = time.time()
    lam_each = lam[:,0]
    mu_each = mu[:,0]
    lam_total = np.sum(lam_each)
    p = lam / np.sum(lam, axis=0, keepdims=True)
    p_each = p[:,0]
    rho_each = lam_each / mu_each
    rho_total = np.sum(rho_each)

    print(time.time()-start)
    n_cycles = args.n_cycles

    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, n_cycles, start, p=CoE_p, policy=args.policy, record = {'num_packet': True, 'cycle_len':True, 'response_sum':True, 'prob_sum':True, 'cross_entropy':True, 'max_response':True, 'reason':True})
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, CE, waits = result['cycle_len'], result['response_sum'], result['prob_sum'], result['reason'], result['num_packet'], result['cross_entropy'], result['max_response']

    print(time.time()-start)

    lam1, lam2 = lam_each
    mu1, mu2 = mu_each
    rho = rho_total
    rho1, rho2 = rho_each

    print(time.time()-start)
    print('Statistics:')
    cycle_num = [cycle_num[0] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[0] for cycle_sum in cycle_sums]

            
    ES, ES2 = np.sum(p_each/mu_each), 2*np.sum(p_each/mu_each**2)
    Se = ES2 / ES / 2
    mean, halfCI = mean_CI(cycle_num)
    print(f'Mean response time in flow 1 (Denominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    print(f'Mean response time in flow 1 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    print(f'Mean response time in flow 1: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {rho*Se/(1-rho1)+1/mu1:.14g} s)')

    for i, gamma in enumerate(threshold[0]):
        cycle_sum_prob = np.array([cycle_sum_prob[0][i] for cycle_sum_prob in cycle_sum_probs])
        # cycle_sum_prob = packets_info[:, 4+i]
        mean, halfCI = mean_CI(cycle_sum_prob)
        print(f'Prob {gamma} in flow 1 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        if 0 == args.test_flow and gamma == threshold[0][-1]:
            RE = halfCI/1.96/mean
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        print(f'Prob {gamma} in flow 1: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[0][i] for reason in reasons])
        # freq = packets_info[:, 4+len(threshold[0])+len(threshold[1])+i]
        print(f'Mean flow 1 {np.mean(cycle_sum_prob[freq==0])}')
        print(f'ratio flow 1 {sum(freq==0)/len(freq)}')
        print(f'Mean flow 2 {np.mean(cycle_sum_prob[freq==1])}')
        print(f'ratio flow 2 {sum(freq==1)/len(freq)}')
        freq = freq[freq>=0]
        mean, halfCI = mean_CI(freq)
        print(f'Frquency reason {gamma} in flow 1 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    # reason1 = packets_npy[:,0][packets_npy[:,0] != np.array(None)]
    # mean, halfCI = mean_CI(reason1)
    # print(f'Percentage due to flow 2: {mean:.4g}  and CI [{mean-halfCI:.4g},{mean+halfCI:.4g}]')


    cycle_num = [cycle_num[1] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[1] for cycle_sum in cycle_sums]
    # cycle_sum_prob = packets_info[:, 5]
    # reasons = [info[3][1] for info in packets_info]    
    mean, halfCI = mean_CI(cycle_num)
    print(f'Mean response time in flow 2 (Denominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    print(f'Mean response time in flow 2 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    print(f'Mean response time in flow 2: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {rho*Se/(1-rho1)/(1-rho)+1/mu2:.14g} s) ')
    for i, gamma in enumerate(threshold[1]):
        cycle_sum_prob = np.array([cycle_sum_prob[1][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        print(f'Prob {gamma} in flow 2 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        print(f'std {np.std(cycle_sum_prob)}')
        if 1 == args.test_flow and gamma == threshold[1][-1]:
            RE = halfCI/1.96/mean
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        print(f'Prob {gamma} in flow 2: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[1][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)
        print(f'Mean flow 1 {np.mean(cycle_sum_prob[freq==0])}')
        print(f'Std flow 1 {np.std(cycle_sum_prob[freq==0])}')
        print(f'ratio flow 1 {sum(freq==0)/len(freq)}')
        print(f'Mean flow 2 {np.mean(cycle_sum_prob[freq==1])}')
        print(f'Std flow 2 {np.std(cycle_sum_prob[freq==1])}')
        print(f'ratio flow 2 {sum(freq==1)/len(freq)}')
        freq = freq[freq>=0]
        mean, halfCI = mean_CI(freq)
        print(f'Frquency reason {gamma} in flow 1 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    print(time.time()-start)
    print(sum(nums))

    CE = np.mean(np.array(CE), axis=0).reshape(4, -1)
    arrival_num, arrival_sum, service_num, service_sum = CE
    print('arrival', lam[:,1], arrival_num / arrival_sum)
    print('service', mu[:,1], service_num / service_sum)
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, waits, RE, sum(nums)

if __name__ == '__main__':
    phase = 0
    rho = args.rho
    lam = lam_ref[:, [0, phase]].copy()
    mu = mu_ref[:, [0, phase]].copy()
    infos = []
    for i in range(args.num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        waits = main(lam.copy(), mu.copy(), port_rate, threshold, args, mix_p)[2]
        gamma = min(args.threshold, np.quantile(waits, 1-rho))
        info.append(gamma)
        threshold[args.test_flow][-1] = gamma
        lam, mu, wait, RE, n_packet = main(lam.copy(), mu.copy(), port_rate, threshold, args, mix_p)
        info.append(RE)
        info.append(n_packet)
        infos.append(info)
    args.n_cycles = args.test_cycles
    info = list(lam[:,1]) + list(mu[:,1])
    info.append(gamma)
    np.random.seed(42)
    lam, mu, wait, RE, n_packet = main(lam.copy(), mu.copy(), port_rate, threshold, args, mix_p)
    info.append(RE)
    info.append(n_packet)
    infos.append(info)
    df = pd.DataFrame(np.array(infos), columns=['lam1', 'lam2', 'mu1', 'mu2', 'gamma', 'RE', 'n_packets'])
    print(df)
    df.to_csv(f'./result/3_cross_entropy/{args.file_name}_t{args.test_flow}_T{args.threshold}_r{args.rho}_iter{args.num_runs}_policy{args.policy}.csv')
    
