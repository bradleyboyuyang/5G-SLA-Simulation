import time

import numpy as np
import pandas as pd

from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_sta1


def CE_update(lam, mu, num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy):
    start = time.time()
    result = static_priority_sim(num_type, lam, mu, threshold, test_flow, port_rate, ExponDistri, cpu_nums, n_cycles, start, policy=policy, mute=True)
    _, __, cycle_sum_probs, ___, nums, CE, waits = result
    cycle_sum_prob = np.array([cycle_sum_prob1[test_flow][-1] for cycle_sum_prob1 in cycle_sum_probs])
    mean, halfCI = cycle_sta1(cycle_len[0], cycle_len[1]**2, cycle_sum_prob)
    CE = np.mean(np.array(CE), axis=0).reshape(4, -1)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, waits, sum(nums), mean, halfCI, time.time()-start

def CE_main(num_type, lam_ref, mu_ref, cycle_length, port_rate, rho, test_flow, cpu_nums, n_cycles, policy, tile, num_runs, test_cycles, file_name):
    start_loacl = time.time()
    print(f'(+{time.time()-start_loacl:.2f}s)Run Cross-Entropy Method...')
    # initial change of measure
    initial_phase = 0
    lam = lam_ref[:, [0, initial_phase]].copy()
    mu = mu_ref[:, [0, initial_phase]].copy()

    infos = []
    cycle_len = cycle_length[test_flow]
    threshold = [[0] for _ in range(num_type)]
    CONTINUE = True 
    gamma_u, gamma_l = 0, 0
    print(f'(+{time.time()-start_loacl:.2f}s)Update and increase the gamma using {n_cycles} cycles...')
    while CONTINUE:
        info = list(lam[:,1]) + list(mu[:,1])
        seed = np.random.randint(100000)
        # Adaptive updating of $\gamma_{t}$
        np.random.seed(seed)
        _, _, waits, _, _, _, time_iter1 = CE_update(lam.copy(), mu.copy(), num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy)
        gamma = np.quantile(waits, 1-rho)
        threshold[test_flow][-1] = gamma
        # Adaptive updating of $\theta_t$.
        np.random.seed(seed)
        lam, mu, _, n_packet, mean, halfCI, time_iter2 = CE_update(lam.copy(), mu.copy(), num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy)
        info.append(threshold[test_flow][-1])
        if mean + halfCI < 1 - tile:
            gamma_u = threshold[test_flow][-1]
            CONTINUE = False
        else:
            gamma_l = threshold[test_flow][-1]
        RE = halfCI / 1.96 / mean
        info.extend([mean, mean-halfCI, mean+halfCI, RE, n_packet, time_iter1+time_iter2])
        infos.append(info)
    if num_runs > 0:
        print(f'(+{time.time()-start_loacl:.2f}s)Fix gamma and update, to stablize the parameter, using {n_cycles} cycles...')
    for _ in range(num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, _, n_packet, mean, halfCI, time_iter = CE_update(lam.copy(), mu.copy(), num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy)
        RE = halfCI / 1.96 / mean
        info.extend([mean, mean-halfCI, mean+halfCI, RE, n_packet, time_iter])
        infos.append(info)
    if test_cycles is not None:
        print(f'(+{time.time()-start_loacl:.2f}s)Evaluate the parameter for {test_cycles} cycles...')
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, _, n_packet,mean, halfCI, time_iter = CE_update(lam.copy(), mu.copy(), num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, test_cycles, policy)
        RE = halfCI / 1.96 / mean
        info.extend([mean, mean-halfCI, mean+halfCI, RE, n_packet, time_iter])
        infos.append(info)
    print(f'(+{time.time()-start_loacl:.2f}s)End the cross-entropy simulation:')
    df = pd.DataFrame(np.array(infos), columns=['lam1', 'lam2', 'mu1', 'mu2', 'gamma', 'prob', 'CI_lower', 'CI_upper', 'RE', 'n_packets', 'time'])
    print(df)
    df.to_csv(f'./result/4_q_tile/B_cross_entropy/{file_name}_t{test_flow}_tile{tile}_r{rho}_policy{policy}.csv')

    return lam, mu, gamma_l, gamma_u


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=5000, help='the size of the training dataset')
    parser.add_argument('-N','--test_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
    parser.add_argument('-t','--test_flow', type=int, default=1, help='the number of cpus')
    parser.add_argument('--tile', type=float, default=0.999, help='threshold')
    parser.add_argument('-k', '--num_runs', type=int, default=0, help='number of runs')
    parser.add_argument('--rho', type=float, default=0.1, help='para of CE')
    parser.add_argument('-f','--file_name', type=str, default='temp')
    parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
    args = parser.parse_args()
    # initial
    # parameters
    num_type = 2
    ## lam mu 
    lam_ref = np.array([[0.6],[0.2]], dtype=float)
    mu_ref = np.array([[2],[1]], dtype=float)
    ## thresholds (must be increasing for each flow)
    
    cycle_length = [[1.49854, 0.01159875913503054, 0.003948999861576336], 
                    [0.50283, 0.005996229095864967, 0.006084164745768508]]
    # initial
    np.random.seed(42)
    start = time.time()
    port_rate = 1 * 8 # 1 byte per second
    res = CE_main(num_type, lam_ref, mu_ref, cycle_length, port_rate, args.rho, args.test_flow, args.cpu_nums, args.n_cycles, args.policy, args.tile, args.num_runs, args.test_cycles, args.file_name)
    print(res)
    print(time.time()-start)
    
