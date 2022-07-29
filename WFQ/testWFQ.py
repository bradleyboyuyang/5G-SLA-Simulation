from pickle import TRUE
import time
import numpy as np
from simulation import static_wfq_sim
from ns.packet.distribution import ExponDistri
import argparse

from utils import cycle_statistics, sample_quantile


parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_cycles', type=int, default=100000, help='')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('-t','--test_flow', type=int, default=0, help='which flow to test')
parser.add_argument('--tile', type=float, default=0.999)
parser.add_argument('--gamma_ref', type=float, default=5)
parser.add_argument('--gamma_p', type=float, default=0.001)
parser.add_argument('--batch_quantile_flag', type=int, default=50)
parser.add_argument('--unbiased_check', type=int, default=1)
parser.add_argument('--file_name', type=str, default='test')
parser.add_argument('--test_case', type=int, default=5)
parser.add_argument('--CoE_phase', type=int, default=0)
args = parser.parse_args()

def main(lam, mu, lam_tilt, mu_tilt, tile, random, batch_quantile_flag=None, idx=None):
    np.random.seed(random.randint(100000))
    L = []
    start = time.time()
    arrival_rates, service_rates = np.zeros((num_type, 2)), np.zeros((num_type, 2))
    arrival_rates[:,0] = lam 
    arrival_rates[:,1] = lam 
    service_rates[:,0] = mu 
    service_rates[:,1] = mu

    # regenerative and not-regenerative
    result = static_wfq_sim(num_type, arrival_rates, service_rates, args.test_flow, args.gamma_ref, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"prob_sum":True, 'save_response':True, 'reason': True, "cycle_len":True, "response_sum":True, 'num_packet':True, "response":True})
    numerator = result['response_sum']
    cycle_len = result['cycle_len']

    # not regenerative
    # num = result['response']
    # # print(len(num))
    # numerator = [i[0] for i in num[0]]
    # print("number of packets: %s"%sum(result['num_packet']))
    # mean, std = np.mean(numerator), np.std(numerator)/np.sqrt(len(numerator))
    # print(f'mean {mean}, mean-halfCI {mean-1.96*std}, mean+halfCI {mean+1.96*std}')

    # regenerative
    # mean, halfCI = cycle_statistics(cycle_len, numerator)
    # print("number of packets: %s"%sum(result['num_packet']))
    # print(f'mean {mean}, mean-halfCI {mean-halfCI}, mean+halfCI {mean+halfCI}')


    # importance sampling
    # compute cycle length
    cycle_len = static_wfq_sim(num_type, arrival_rates, service_rates, args.test_flow, 100, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"cycle_len":True})['cycle_len']
    mean, std = np.mean(cycle_len), np.std(cycle_len)/np.sqrt(len(cycle_len))
    print(f'mean {mean}, std {std}, RE {std/mean}')

    L.extend([len(cycle_len), mean, std, std/mean])


    # compute numerator
    np.random.seed(random.randint(100000))
    arrival_rates, service_rates = np.zeros((num_type, 2)), np.zeros((num_type, 2))
    arrival_rates[:,0] = lam 
    arrival_rates[:,1] = lam_tilt
    service_rates[:,0] = mu 
    service_rates[:,1] = mu_tilt
    result = static_wfq_sim(num_type, arrival_rates, service_rates, args.test_flow, args.gamma_ref, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"prob_sum":True, 'save_response':True, 'reason': True, "cycle_len":True, "response_sum":True, 'num_packet':True, "response":True})
    prob_numerator = result['prob_sum']
    resp_numerator = result['response_sum']

    # steady-state response time expectation
    mean, halfCI = cycle_statistics(cycle_len, resp_numerator)
    print("number of packets: %s"%sum(result['num_packet']))
    print(f'mean {mean}, mean-halfCI {mean-halfCI}, mean+halfCI {mean+halfCI}')

    # probability numerator
    mean, std = np.mean(prob_numerator), np.std(prob_numerator)/np.sqrt(len(prob_numerator))
    print(f'{args.gamma_ref}: mean {mean}, std {std}, RE {std/mean}')
    L.extend([args.gamma_ref, len(prob_numerator), mean, std, std/mean])

    # probability
    mean, CI = cycle_statistics(cycle_num=cycle_len, cycle_sum=prob_numerator)
    print(f'{args.gamma_ref}: mean {mean}, CI ({mean-CI}, {mean+CI}), RE {CI/1.96/mean}')
    L.extend([args.gamma_ref, mean, mean-CI, mean+CI, CI/1.96/mean])
    

if __name__ == '__main__':
    random = np.random.RandomState(42)
    num_type = 2
    lam = [0.2, 0.4]
    mu = [2, 1]
    weights = {0:2, 1:1}

    lam_tilt = lam
    mu_tilt = mu        
      
    for idx in range(args.unbiased_check):
        main(lam, mu, lam_tilt, mu_tilt, args.tile, random, args.batch_quantile_flag, idx)
