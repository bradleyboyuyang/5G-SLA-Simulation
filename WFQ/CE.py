import argparse
import time

import numpy as np
import pandas as pd

from ns.packet.distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_statistics

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_cycles', type=int, default=100000, help='')
parser.add_argument('--CE_cycles', type=int, default=10000, help='')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('-t','--test_flow', type=int, default=1, help='the number of cpus')
parser.add_argument('--tile', type=float, default=0.999)
parser.add_argument('--gamma_ref', type=float, default=24.368)
parser.add_argument('--gamma_p', type=float, default=0.001)
parser.add_argument('--batch_quantile_flag', type=int, default=None)
parser.add_argument('--unbiased_check', type=int, default=100)
parser.add_argument('--stable_iter', type=int, default=10)
args = parser.parse_args()

def CE_update(lam, mu, num_type, port_rate, gamma, test_flow, cycle_num, cpu_nums, n_cycles, rho, start, gamma_ref=None):
    start = time.time()
    # simulate
    result = static_priority_sim(num_type, lam, mu, test_flow, gamma, port_rate, ExponDistri, cpu_nums, n_cycles, start, True, {'max_response':True, 'save_response':True, 'cross_entropy':True})
    # get the quantile of max_response time
    gamma = np.quantile(result['max_response'], 1-rho)
    if gamma_ref is not None:
        gamma = min(gamma_ref, gamma)
    # compute the probability for the new gamma
    cycle_sum_prob = np.array([sum([(packet[0]>gamma)*np.exp(packet[1]) for packet in packet_cycle]) for packet_cycle in result['response']])
    mean, halfCI = cycle_statistics(cycle_num=cycle_num, cycle_sum=cycle_sum_prob)
    CE_para = np.array(result['cross_entropy'])
    CE = CE_para * cycle_sum_prob.reshape(-1,1,1)
    CE = np.mean(CE, axis=0)
    assert CE.shape[0] == 4
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, gamma, mean, halfCI, time.time()-start

def CE_stable(lam, mu, num_type, port_rate, gamma, test_flow, cycle_num, cpu_nums, n_cycles):
    start = time.time()
    # simulate
    result = static_priority_sim(num_type, lam, mu, test_flow, gamma, port_rate, ExponDistri, cpu_nums, n_cycles, start, True, {'prob_sum':True, 'num_packet':True, 'cross_entropy':True})
    cycle_sum_prob = np.array(result['prob_sum'])
    mean, halfCI = cycle_statistics(cycle_num=cycle_num, cycle_sum=cycle_sum_prob)
    CE = np.mean(np.array(result['cross_entropy']), axis=0)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, mean, halfCI, time.time()-start
###############
def CE_main(num_type, lam_ref, mu_ref, cycle_num, port_rate, rho, test_flow, cpu_nums, n_cycles, tile, num_runs, test_cycles, file_name, gamma_ref=None):
    start_loacl = time.time()
    print(f'(+{time.time()-start_loacl:.2f}s)Run Cross-Entropy Method...')
    # initial change of measure
    initial_phase = 0
    lam = lam_ref[:, [0, initial_phase]].copy()
    mu = mu_ref[:, [0, initial_phase]].copy()

    infos = []
    CONTINUE = True 
    gamma_u, gamma_l, gamma = 0, 0, 0
    print(f'(+{time.time()-start_loacl:.2f}s)Update and increase the gamma using {n_cycles} cycles...')
    while CONTINUE:
        info = list(lam[:,1]) + list(mu[:,1])
        # update parameters
        lam, mu, gamma, mean, halfCI, time_iter = CE_update(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, cpu_nums, n_cycles, rho, start_loacl, gamma_ref=gamma_ref)
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
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, time_iter])
        infos.append(info)
    print(f'(+{time.time()-start_loacl:.2f}s)Fix gamma and update, to stablize the parameter, using {n_cycles} cycles...')
    for _ in range(num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, mean, halfCI, time_iter = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, cpu_nums, n_cycles)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, time_iter])
        infos.append(info)
    # save last one
    info = list(lam[:,1]) + list(mu[:,1])
    info.append(gamma)
    info.extend([None, None, None])
    infos.append(info)
    print(f'(+{time.time()-start_loacl:.2f}s)End the cross-entropy simulation:')
    df = pd.DataFrame(np.array(infos), columns=[f'lam{k+1}' for k in range(num_type)]+[f'mu{k+1}' for k in range(num_type)]+['gamma', 'prob', 'RE', 'time'])
    print(df)
    df.to_csv(f'./results/cross_entropy/{file_name}_t{test_flow}_tile{tile}_r{rho}_K{num_type}.csv')
    print('where the last line is the final result.')

    return lam, mu, gamma_l, gamma_u


def main(lam, mu, tile, random, rho):
    start = time.time()
    print(f'({time.time()-start:.2f}s) Naive Simulation for the denominators under {args.n_cycles} cycles...')
    np.random.seed(random.randint(100000))
    L = []
    start = time.time()
    arrival_rates, service_rates = np.zeros((num_type, 2)), np.zeros((num_type, 2))
    arrival_rates[:,0] = lam 
    arrival_rates[:,1] = lam 
    service_rates[:,0] = mu 
    service_rates[:,1] = mu
    cycle_len = static_priority_sim(num_type, arrival_rates, service_rates, args.test_flow, 100, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"cycle_len":True})['cycle_len']
    mean, std = np.mean(cycle_len), np.std(cycle_len)/np.sqrt(len(cycle_len))
    print(f'mean {mean}, std {std}, RE {std/mean}')
    L.extend([len(cycle_len), mean, std, std/mean])
    print(f'({time.time()-start:.2f}s) Cross Entropy for the nominators under {args.CE_cycles} cycles...')
    np.random.seed(random.randint(100000))
    arrival_rates, service_rates = np.zeros((num_type, 2)), np.zeros((num_type, 2))
    arrival_rates[:,0] = lam 
    arrival_rates[:,1] = lam 
    service_rates[:,0] = mu 
    service_rates[:,1] = mu
    
    res = CE_main(num_type, arrival_rates, service_rates, cycle_len, 8, rho, args.test_flow, args.cpu_nums, args.CE_cycles, tile, args.stable_iter, None, 'test', gamma_ref=args.gamma_ref)
    print(f'({time.time()-start:.2f}s) End Simulation.')
    return res

if __name__ == '__main__':
    random = np.random.RandomState(42)
    # num_type = 4
    # lam = [0.15, 0.15, 0.15, 0.15]
    # mu = [1, 1, 1, 1]

    num_type = 2
    lam = [0.2, 0.1]
    mu = [1.0, 1.0]
    if args.test_flow == 0:
        args.gamma_ref = 9.0717
    elif args.test_flow == 1:
        args.gamma_ref =  13.9981

    L = main(lam, mu, args.tile, random, 0.1)
    # import csv 
    # with open(f'./tail_prob_K{num_type}_flow{args.test_flow}_quantile.csv', "a") as f:
    #     writer = csv.writer(f)
    #     writer.writerows([L])