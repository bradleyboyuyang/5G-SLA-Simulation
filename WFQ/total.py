import argparse
import time

import numpy as np
from cross_entropy import CE_main, CE_up, CE_up_down
from cross_entropy_denominator import CE_nominator

from ns.packet.distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_statistics, quantile_CI, sample_quantile

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_cycles', type=int, default=100000, help='')
parser.add_argument('--CE_cycles', type=int, default=10000, help='')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('-t','--test_flow', type=int, default=0, help='the number of cpus')
parser.add_argument('--tile', type=float, default=0.999)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--gamma_ref', type=float, default=None)
parser.add_argument('--gamma_p', type=float, default=0.001)
parser.add_argument('--batch_quantile_flag', type=int, default=50)
parser.add_argument('--unbiased_check', type=int, default=100)
parser.add_argument('--stable_iter', type=int, default=5)
parser.add_argument('--IS4denominator', type=int, default=0)
args = parser.parse_args()




def main(lam_original, mu_original, tile, random, rho, batch_quantile_flag):
    start = time.time()
    print(f'({time.time()-start:.2f}s) Naive Simulation for the denominators under {args.n_cycles} cycles...')
    np.random.seed(random.randint(100000))
    L = []
    start = time.time()
    arrival_rates, service_rates = np.zeros((num_type, 2)), np.zeros((num_type, 2))
    arrival_rates[:,0] = lam_original
    arrival_rates[:,1] = lam_original
    service_rates[:,0] = mu_original
    service_rates[:,1] = mu_original
    if args.IS4denominator:
        arrival_rates, service_rates = CE_nominator(num_type, arrival_rates, service_rates, 8, rho, args.test_flow, args.cpu_nums, args.CE_cycles, 1, 'test')
    cycle_len = static_priority_sim(num_type, arrival_rates, service_rates, args.test_flow, 100, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"cycle_len":True})['cycle_len']
    mean, std = np.mean(cycle_len), np.std(cycle_len)/np.sqrt(len(cycle_len))
    print(f'mean {mean}, std {std}, RE {std/mean}')
    L.extend([len(cycle_len), mean, std, std/mean])
    lam_de, mu_de = arrival_rates, service_rates
    print(f'({time.time()-start:.2f}s) Cross Entropy for the nominators under {args.CE_cycles} cycles...')
    np.random.seed(random.randint(100000))
    arrival_rates, service_rates = np.zeros((num_type, 2)), np.zeros((num_type, 2))
    arrival_rates[:,0] = lam_original 
    arrival_rates[:,1] = lam_original
    service_rates[:,0] = mu_original
    service_rates[:,1] = mu_original
    
    # lam, mu, gamma_l, gamma_u, gamma = CE_main(num_type, arrival_rates, service_rates, cycle_len, 8, rho, args.test_flow, args.cpu_nums, args.CE_cycles, tile, args.stable_iter, None, 'test', gamma_ref=args.gamma_ref)
    # lam, mu, gamma_l, gamma_u, gamma = CE_up_down(num_type, arrival_rates, service_rates, cycle_len, 8, rho, args.test_flow, args.cpu_nums, args.CE_cycles, tile, args.stable_iter, None, 'test', gamma_ref=args.gamma_ref)
    lam, mu, gamma_l, gamma_u, gamma = CE_up(num_type, arrival_rates, service_rates, cycle_len, 8, rho, args.test_flow, args.cpu_nums, args.CE_cycles, tile, args.stable_iter, None, 'CE_de', gamma_ref=args.gamma_ref)
    
    print(f'({time.time()-start:.2f}s) Quantile Estimation {args.n_cycles} cycles...')
    np.random.seed(random.randint(100000))
    arrival_rates, service_rates = lam, mu
    args.gamma_ref = gamma
    result = static_priority_sim(num_type, arrival_rates, service_rates, args.test_flow, args.gamma_ref, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"prob_sum":True, 'save_response':True})
    nominator = result['prob_sum']
    mean, std = np.mean(nominator), np.std(nominator)/np.sqrt(len(nominator))
    print(f'{args.gamma_ref}: mean {mean}, std {std}, RE {std/mean}')
    L.extend([args.gamma_ref, len(nominator), mean, std, std/mean])
    mean, CI = cycle_statistics(cycle_num=cycle_len, cycle_sum=nominator)
    print(f'{args.gamma_ref}: mean {mean}, CI ({mean-CI}, {mean+CI}), RE {CI/1.96/mean}')
    L.extend([args.gamma_ref, mean, mean-CI, mean+CI, CI/1.96/mean])
    if args.gamma_p is not None:
        L.append((mean-CI<args.gamma_p)*(args.gamma_p<mean+CI))
    
    Q2_total = sample_quantile(result['response'], cycle_len, tile)
    print(f'sample {tile} quantile is {Q2_total}')
    Q2_total, section_std, batch_mean, batch_std = quantile_CI(cycle_len, result['response'], batch_quantile_flag, tile, Q2_total, random)
    
    L.extend([Q2_total, Q2_total-1.96*section_std, Q2_total+1.96*section_std, section_std/Q2_total])
    if args.gamma_p is not None:
        L.append((Q2_total-1.96*section_std<args.gamma_ref)*(args.gamma_ref<Q2_total+1.96*section_std))
        
    print(f'({time.time()-start:.2f}s) Tail Probability Estimation {args.n_cycles} cycles...')
    np.random.seed(random.randint(100000))
    cycle_len = static_priority_sim(num_type, lam_de, mu_de, args.test_flow, 100, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"cycle_len":True})['cycle_len']
    mean, std = np.mean(cycle_len), np.std(cycle_len)/np.sqrt(len(cycle_len))
    print(f'mean {mean}, std {std}, RE {std/mean}')
    arrival_rates, service_rates = lam, mu
    args.gamma_ref = Q2_total
    result = static_priority_sim(num_type, arrival_rates, service_rates, args.test_flow, args.gamma_ref, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"prob_sum":True, 'save_response':True, 'reason': True})
    nominator = result['prob_sum']
    mean, std = np.mean(nominator), np.std(nominator)/np.sqrt(len(nominator))
    print(f'{args.gamma_ref}: mean {mean}, std {std}, RE {std/mean}')
    L.extend([args.gamma_ref, len(nominator), mean, std, std/mean])
    mean, CI = cycle_statistics(cycle_num=cycle_len, cycle_sum=nominator)
    print(f'{args.gamma_ref}: mean {mean}, CI ({mean-CI}, {mean+CI}), RE {CI/1.96/mean}')
    L.extend([args.gamma_ref, mean, mean-CI, mean+CI, CI/1.96/mean])
    if args.gamma_p is not None:
        L.append((mean-CI<args.gamma_p)*(args.gamma_p<mean+CI))
    reason = np.array(result['reason'])
    L.extend([np.mean(reason==0), np.mean(reason==1), np.mean(reason==100)])
    print(f'event ratio: class 1 {np.mean(reason==0)}, class 2 {np.mean(reason==1)}, init {np.mean(reason==100)}, cond {np.sum(reason==1)/(np.sum(reason>=0))}')
    Q2_total = sample_quantile(result['response'], cycle_len, tile)
    print(f'sample {tile} quantile is {Q2_total}')
    Q2_total, section_std, batch_mean, batch_std = quantile_CI(cycle_len, result['response'], batch_quantile_flag, tile, Q2_total, random)
    print(f'({time.time()-start:.2f}s) End Simulation.')
    return L
    

if __name__ == '__main__':
    
    
    # num_type = 4
    # lam = [0.15, 0.15, 0.15, 0.15]
    # mu = [1, 1, 1, 1]
    
    # num_type = 8
    # lam = [0.01,0.01,0.01,0.4,0.2,0.2,0.1,0.45]
    # mu = 1/np.array([0.375,0.375,0.375,0.375,0.375,0.375,0.375,0.375])

    num_type = 2
    lam = [0.1, 0.2]
    mu = [1, 1]
    
    # num_type = 2
    # lam = [0.2, 0.4]
    # mu = [2, 1]

    # num_type = 2
    # lam = [0.6, 0.2]
    # mu = [2, 1]
    args.gamma_ref = None
    random = np.random.RandomState(26871)
    main(lam, mu, args.tile, random, args.rho, args.batch_quantile_flag)
    # import csv 
    # with open(f'./tail_prob_K{num_type}_flow{args.test_flow}_quantile.csv', "a") as f:
    #     writer = csv.writer(f)
    #     writer.writerows([L])