import time

import numpy as np
import pandas as pd

from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_sta1


def CE_update(lam, mu, num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy, rho, gamma_ref=None):
    start = time.time()
    # simulate
    result = static_priority_sim(num_type, lam, mu, threshold, test_flow, port_rate, ExponDistri, cpu_nums, n_cycles, start, policy=policy, mute=True, record={'max_response':True, 'save_response':True, 'num_packet':True})
    # get the quantile of max_response time
    gamma = np.quantile(result['max_response'], 1-rho)
    if gamma_ref is not None:
        gamma = min(gamma_ref, gamma)
    # compute the probability for the new gamma
    cycle_sum_prob = np.array([sum([(packet[0]>gamma)*np.exp(packet[1]) for packet in packet_cycle])
                               for packet_cycle in result['response']])
    mean, halfCI = cycle_sta1(cycle_len[0], cycle_len[1]**2, cycle_sum_prob)
    CE_para = np.array(result['CE'])
    CE = CE_para * cycle_sum_prob.reshape(-1,1,1)
    CE = np.mean(CE, axis=0).reshape(4, -1)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, gamma, sum(result['num_packet']), mean, halfCI, time.time()-start

def CE_stable(lam, mu, num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy):
    start = time.time()
    # simulate
    result = static_priority_sim(num_type, lam, mu, threshold, test_flow, port_rate, ExponDistri, cpu_nums, n_cycles, start, policy=policy, mute=True, record={'prob_sum':True, 'num_packet':True, 'cross_entropy':True})
    cycle_sum_prob = np.array([cycle_sum_prob1[test_flow][-1] for cycle_sum_prob1 in result['prob_sum']])
    mean, halfCI = cycle_sta1(cycle_len[0], cycle_len[1]**2, cycle_sum_prob)
    CE = np.mean(np.array(result['cross_entropy']), axis=0).reshape(4, -1)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, sum(result['num_packet']), mean, halfCI, time.time()-start

def CE_main(num_type, lam_ref, mu_ref, cycle_length, port_rate, rho, test_flow, cpu_nums, n_cycles, policy, tile, num_runs, test_cycles, file_name, gamma_ref=None):
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
        # update parameters
        lam, mu, gamma, n_packet, mean, halfCI, time_iter = CE_update(lam.copy(), mu.copy(), num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy, rho, gamma_ref=gamma_ref)
        info.append(gamma)
        if gamma_ref is not None:
            if np.isclose(gamma_ref, gamma):
                CONTINUE = False
        else:
            if mean + halfCI < 1 - tile:
                gamma_u = gamma
                CONTINUE = False
            elif mean - halfCI > 1 - tile:
                gamma_l = gamma
        threshold[test_flow][-1] = gamma
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, n_packet, time_iter])
        infos.append(info)
    print(f'(+{time.time()-start_loacl:.2f}s)Fix gamma and update, to stablize the parameter, using {n_cycles} cycles...')
    for _ in range(num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, n_packet, mean, halfCI, time_iter = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, n_cycles, policy)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, n_packet, time_iter])
        infos.append(info)
    if test_cycles is not None:
        print(f'(+{time.time()-start_loacl:.2f}s)Evaluate the parameter for {test_cycles} cycles...')
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, n_packet,mean, halfCI, time_iter = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, threshold, test_flow, cycle_len, cpu_nums, test_cycles, policy)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, n_packet, time_iter])
        infos.append(info)
    print(f'(+{time.time()-start_loacl:.2f}s)End the cross-entropy simulation:')
    df = pd.DataFrame(np.array(infos), columns=[f'lam{k+1}' for k in range(num_type)]+[f'mu{k+1}' for k in range(num_type)]+['gamma', 'prob', 'RE', 'n_packets', 'time'])
    print(df)
    df.to_csv(f'./result/4_q_tile/B_cross_entropy/{file_name}_t{test_flow}_tile{tile}_r{rho}_policy{policy}_K{num_type}.csv')

    return lam, mu, gamma_l, gamma_u


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=10000, help='the size of the training dataset')
    parser.add_argument('-N','--test_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
    parser.add_argument('-t','--test_flow', type=int, default=1, help='the number of cpus')
    parser.add_argument('--tile', type=float, default=0.999, help='threshold')
    parser.add_argument('-k', '--num_runs', type=int, default=3, help='number of runs')
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

    # # parameters
    # num_type = 4
    # lam_ref = np.array([0.6, 0.2, 0.2, 0.2], dtype=float).reshape(-1, 1)
    # mu_ref = np.array([2, 1, 2, 2], dtype=float).reshape(-1, 1)
    # cycle_length = [[1.65031, 0.021216967774406934, 0.00655936373067538], 
    #                 [0.54935, 0.008808285745208314, 0.00818061953106762], 
    #                 [0.54745, 0.007939883548261901, 0.007399691285069274], 
    #                 [0.55138, 0.00804674913692291, 0.007445834548826756]]

    # initial
    np.random.seed(196)
    start = time.time()
    port_rate = 1 * 8 # 1 byte per second
    res = CE_main(num_type, lam_ref, mu_ref, cycle_length, port_rate, args.rho, args.test_flow, args.cpu_nums, args.n_cycles, args.policy, args.tile, args.num_runs, args.test_cycles, args.file_name)
    print(res)
    print(time.time()-start)
    
