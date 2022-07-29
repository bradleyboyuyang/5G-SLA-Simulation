import time

import numpy as np
import csv
from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_sta1, mean_CI 


def evaluate(num_type, lam, mu, port_rate, threshold, test_flow, cpu_nums, n_cycles, cycle_len, mix_p, policy, mute=False):
    start_local = time.time()

    print(f'+({time.time()-start_local:.2f}s)Run simulation...')
    result = static_priority_sim(num_type, lam, mu, threshold, test_flow, port_rate, ExponDistri, cpu_nums, n_cycles, start_local, p=mix_p, policy=policy, mute=mute, record = { 'num_packet': True, 'prob_sum': True,'cross_entropy': True})
    cycle_sum_probs = result['prob_sum']
    nums = result['num_packet']
    CE  = result['cross_entropy']

    L = []
    print(f'+({time.time()-start_local:.2f}s)Statistics:')
    for i, gamma in enumerate(threshold[test_flow]):
        cycle_sum_prob = np.array([cycle_sum_prob[test_flow][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = cycle_sta1(cycle_len[0], cycle_len[1]**2, cycle_sum_prob)
        L.append([gamma, mean, halfCI, mean-halfCI, mean+halfCI, halfCI/1.96/mean])

    CE = np.mean(np.array(CE), axis=0).reshape(4, -1)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum

    print(sum(nums))
    return np.array(L), lam.copy(), mu.copy()

def evaluate2(idx, num_type, lam, mu, port_rate, threshold, test_flow, cpu_nums, n_cycles, cycle_len, mix_p, policy, args, start, mute=False):
    start_local = time.time()

    print(f'+({time.time()-start_local:.2f}s)Run simulation...')
    result = static_priority_sim(num_type, lam, mu, threshold, test_flow, port_rate, ExponDistri, cpu_nums, n_cycles, start_local, p=mix_p, policy=policy, mute=mute, record = { 'num_packet': True, 'prob_sum': True,'cross_entropy': True})
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums = result['cycle_len'], result['response_sum'], result['prob_sum'], result['reason'], result['num_packet']
    CE  = result['cross_entropy']

    L = []
    L.extend([idx])

    if mix_p is not None:
        L.extend(list(mix_p.keys()) + list(mix_p.values()))
    for i in range(num_type):
        L.extend(list(lam[i])+list(mu[i]))

    L.extend([args.test_cycles, time.time()-start_local, sum(nums)])

    print(f'+({time.time()-start_local:.2f}s)Statistics:')
    for i, gamma in enumerate(threshold[test_flow]):
        cycle_sum_prob = np.array([cycle_sum_prob[test_flow][i] for cycle_sum_prob in cycle_sum_probs])

        mean, halfCI = mean_CI(cycle_sum_prob)
        L.extend([gamma, mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob (gamma={gamma}) in flow {test_flow} (Numerator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')

        mean, halfCI = cycle_sta1(cycle_len[0], cycle_len[1]**2, cycle_sum_prob)
        L.extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])

        # L.extend([gamma, mean, halfCI, mean-halfCI, mean+halfCI, halfCI/1.96/mean])

    with open(f"result/4_unbias_check/{args.file_name}_t{args.test_flow}_iter{args.unbiased_check}_policy{args.policy}_K{num_type}.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerows([L])

    CE = np.mean(np.array(CE), axis=0).reshape(4, -1)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum

    print(sum(nums))
    return np.array([L]), lam.copy(), mu.copy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
    parser.add_argument('--test_flow', type=int, default=0, help='which flow to swtich measure')
    parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
    args = parser.parse_args()

    start = time.time()
    mix_p = None
    num_type = 2
    threshold = [[8.1681],  
                 [9.8891]]
    # cycle_length = [[1.49824,    0.01170459, 0.00398583],
    #                 [0.5011,     0.00597739, 0.00608598]]
    cycle_length = [[0.62469,    0.00407101, 0.00332492],
                    [0.62577,    0.0040704,  0.00331868]]
    lam = np.array([[0.1,        0.27983031],
                    [0.1,        0.09590162]] , dtype=float)
    mu = np.array([[1.,         0.20767235],
                    [1.,         0.22058204]], dtype=float)
    # lam = np.array([[0.6, 0.6],
    #                 [0.2, 0.2]])
    # mu = np.array([[   2,    2.0],
    #             [   1,    1.0]])
    # lam[:,1] = [0.8204897498205567,0.5791510463854309]
    # mu[:,1] = [1.3366737345090036,0.4373962955149862]
    # initial
    np.random.seed(42)
    
    port_rate = 1 * 8 # 1 byte per second
    res = evaluate(num_type, lam, mu, port_rate, threshold, args.test_flow, args.cpu_nums, args.n_cycles, cycle_length[args.test_flow], mix_p, args.policy)
    print((res))
    print(time.time()-start)
