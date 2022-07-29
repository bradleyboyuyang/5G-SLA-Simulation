import argparse
import time

import numpy as np
import pandas as pd

from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_statistics, mean_CI


def get_flow_result(num_type, flow_id:int, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, args, start):
    """
    target: obtain typical statistics of the events with type "flow_id"
    theory: theoretical waiting time
    """
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]

    # statistics of sojourn time
    mean, halfCI = mean_CI(cycle_num)
    print(f'Mean response time in flow {flow_id+1} (Denominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    print(f'Mean response time in flow {flow_id+1} (Numerator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    print(f'Mean response time in flow {flow_id+1}: {mean:.14g}s  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {theory:.14g}s)')
    print("=====================================================================================================================================")
    
    # statistics of probability
    for i, gamma in enumerate(threshold[flow_id]):
        cycle_sum_prob = np.array([cycle_sum_prob[flow_id][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        print(f'Prob (gamma={gamma}) in flow {flow_id+1} (Numerator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        if flow_id == args.test_flow and gamma == threshold[flow_id][-1]:
            RE = halfCI/1.96/mean
        else:
            RE = 0
        
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        print(f'Prob (gamma={gamma}) in flow {flow_id+1}: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[flow_id][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)

        # supplementary statistics
        for k in range(num_type):
            print(f'Mean flow {k+1}: {np.mean(cycle_sum_prob[freq==k])}')
            print(f'Std flow {k+1}: {np.std(cycle_sum_prob[freq==k])}')
            print(f'ratio flow {k+1}: {sum(freq==k)/len(freq)}')

        freq = freq[freq>=0]
        mean, halfCI = mean_CI(freq)
        print(f'Frquency reason {k+1} in flow  (Numerator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        print(time.time()-start)
        print(sum(nums))
        print("=====================================================================================================================================")
    return RE


def main(lam, mu, port_rate, threshold, args, CoE_p, num_type, start):
    start_local = time.time()
    RE = 0
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

    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, args.n_cycles, start_local, p=CoE_p, policy=args.policy, record = {'num_packet': True, 'cycle_len':True, 'response_sum':True, 'prob_sum':True, 'cross_entropy':True, 'max_response':True, 'reason':True})
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, CE, waits = result['cycle_len'], result['response_sum'], result['prob_sum'], result['reason'], result['num_packet'], result['cross_entropy'], result['max_response']

    # obtain statistics
    print("=====================================================================================================================================")
    print(f'({time.time()-start_local:.2f}s)Statistics:')
    for flow_id in range(num_type):
        # calculate theoretical sojourn time (mean waiting time + service time)
        theory = rho*Se/(1-rho_each[:flow_id+1].sum())/(1-rho_each[:flow_id].sum()) + 1/mu_each[flow_id]
        # obtain simulation result for each type of events
        re = get_flow_result(num_type, flow_id, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, args, start)
        if re != 0:
            RE = re
    print("=====================================================================================================================================")

    CE = np.mean(np.array(CE), axis=0).reshape(4, -1)
    arrival_num, arrival_sum, service_num, service_sum = CE
    print('arrival', lam[:,1], arrival_num / arrival_sum)
    print('service', mu[:,1], service_num / service_sum)
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, waits, RE, sum(nums)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=10000, help='the size of the training dataset')
    parser.add_argument('-N','--test_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
    parser.add_argument('-t','--test_flow', type=int, default=4, help='which flow to switch measure')
    parser.add_argument('-T','--threshold', type=float, default=10, help='threshold')
    parser.add_argument('-k', '--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--rho', type=int, default=0.1, help='para of CE')
    parser.add_argument('-f','--file_name', type=str, default='temp')
    parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
    args = parser.parse_args()

    phase = 0

    # # parameters
    num_type = 8
    lam_ref = np.array([[0.1],[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]], dtype=float)
    mu_ref = np.array([[1],[1], [1], [1], [1], [1], [1], [1]], dtype=float)
    ## thresholds (must be increasing for each flow)
    threshold = [[10] for _ in range(num_type)]
    

    rho = args.rho
    lam = lam_ref[:, [0, phase]].copy()
    mu = mu_ref[:, [0, phase]].copy()

    # decide which set of CE parameters to get (CE for which type of customers, and for which threshold)
    threshold[args.test_flow].append(args.threshold)

    # initial
    np.random.seed(42)
    mix_p = None
    start = time.time()
    port_rate = 1 * 8 # 1 byte per second

    infos = []

    # for i in range(args.num_runs):
    gamma = 0
    trial = 0
    while (gamma < args.threshold) & (trial < args.num_runs):
        trial += 1
        info = list(lam[:,1]) + list(mu[:,1])
        waits = main(lam.copy(), mu.copy(), port_rate, threshold, args, mix_p, num_type, start)[2]
        gamma = min(args.threshold, np.quantile(waits, 1-rho))
        info.append(gamma)
        threshold[args.test_flow][-1] = gamma
        lam, mu, wait, RE, n_packet = main(lam.copy(), mu.copy(), port_rate, threshold, args, mix_p, num_type, start)
        info.append(RE)
        info.append(n_packet)
        infos.append(info)

    # 更改args的参数
    args.n_cycles = args.test_cycles
    info = list(lam[:,1]) + list(mu[:,1])
    info.append(gamma)
    np.random.seed(42)
    lam, mu, wait, RE, n_packet = main(lam.copy(), mu.copy(), port_rate, threshold, args, mix_p, num_type, start)
    info.append(RE)
    info.append(n_packet)
    infos.append(info)
    df = pd.DataFrame(np.array(infos), columns=[f'lam{k+1}' for k in range(num_type)]+[f'mu{k+1}' for k in range(num_type)]+['gamma', 'RE', 'n_packets'])  
    # df = pd.DataFrame(np.array(infos), columns=['lam1', 'lam2', 'mu1', 'mu2', 'gamma', 'RE', 'n_packets'])
      
    print(df)
    df.to_csv(f'./result/3_cross_entropy/{args.file_name}_t{args.test_flow}_T{args.threshold}_r{args.rho}_iter{args.num_runs}_policy{args.policy}_K{num_type}.csv')
    
