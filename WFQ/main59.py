import time
import numpy as np
from simulation import static_priority_sim
from ns.packet.distribution import ExponDistri
import argparse

from utils import cycle_statistics, sample_quantile




parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_cycles', type=int, default=100000, help='')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('-t','--test_flow', type=int, default=0, help='the number of cpus')
parser.add_argument('--tile', type=float, default=0.99999)
parser.add_argument('--gamma_ref', type=float, default=None)
parser.add_argument('--gamma_p', type=float, default=0.00001)
parser.add_argument('--batch_quantile_flag', type=int, default=50)
parser.add_argument('--unbiased_check', type=int, default=100)
parser.add_argument('--file_name', type=str, default='test')
parser.add_argument('--test_case', type=int, default=2)
parser.add_argument('--CoE_phase', type=int, default=2)
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
    cycle_len = static_priority_sim(num_type, arrival_rates, service_rates, args.test_flow, 100, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"cycle_len":True})['cycle_len']
    mean, std = np.mean(cycle_len), np.std(cycle_len)/np.sqrt(len(cycle_len))
    print(f'mean {mean}, std {std}, RE {std/mean}')
    # np.save(f'results/info/{args.file_name}_{idx}_cycle.npy',cycle_len)
    L.extend([len(cycle_len), mean, std, std/mean])
    
    np.random.seed(random.randint(100000))
    arrival_rates, service_rates = np.zeros((num_type, 2)), np.zeros((num_type, 2))
    arrival_rates[:,0] = lam 
    arrival_rates[:,1] = lam_tilt
    service_rates[:,0] = mu 
    service_rates[:,1] = mu_tilt
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
    n_result = len(result['response'])
    # np.save(f'results/info/{args.file_name}_{idx}_response.npy', result['response'], allow_pickle=True)
    while True:
        np.random.seed(random.randint(100000))
        np.random.shuffle(cycle_len)
        if batch_quantile_flag is None:
            batch_quantile = input('batch size:')
        else:
            batch_quantile = batch_quantile_flag
        if batch_quantile:
            batch_quantile = int(batch_quantile)
            L1 = []
            r = 0
            for index in range(0, n_result, batch_quantile):
                r += 1
                batch = result['response'][index:min(index+batch_quantile,n_result)]
                batch_tau = cycle_len[index:min(index+batch_quantile,n_result)]
                Q2 = sample_quantile(batch, batch_tau, tile)
                L1.append(Q2)
            batch_mean = np.mean(L1)
            batch_std = np.std(L1)/ np.sqrt(r)
            section_std = np.sqrt(np.mean((np.array(L1)-Q2_total)**2)) / np.sqrt(r)
            print(f'{Q2_total} sectioning CI ({Q2_total-1.96*section_std}, {Q2_total+1.96*section_std}) RE: {section_std/Q2_total}')
            print(f'{batch_mean} batching CI ({batch_mean-1.96*batch_std}, {batch_mean+1.96*batch_std}) RE: {batch_std/batch_mean}')
        else:
            break
        if batch_quantile_flag is not None:
            break
    L.extend([Q2_total, Q2_total-1.96*section_std, Q2_total+1.96*section_std, section_std/Q2_total])
    if args.gamma_p is not None:
        L.append((Q2_total-1.96*section_std<args.gamma_ref)*(args.gamma_ref<Q2_total+1.96*section_std))
    del result
    print(f'time: {time.time()-start}')
    return L

if __name__ == '__main__':
    random = np.random.RandomState(2022)
    # num_type = 4
    # lam = [0.15, 0.15, 0.15, 0.15]
    # mu = [1, 1, 1, 1]
    # lam_tilt = [0.3918284534863452, 0.1917008884432369, 0.19384927003983013, 0.19015014095684432]
    # mu_tilt = [0.30289029745149215, 0.4324399899548464, 0.39608747332020994, 0.34511310915205345]
    
    num_type = 2
    if args.test_case == 2:
        lam = [0.1, 0.2]
        mu = [1.0, 1.0]
        if args.test_flow == 0:
            args.gamma_ref =  13.8090
        elif args.test_flow == 1:
            args.gamma_ref =  20.6368
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        elif args.CoE_phase == 1:
            lam_tilt =  [1/3, 2/3]
            mu_tilt = [0.3, 0.3]
        elif args.CoE_phase == 2:
            if args.test_flow == 0:
                lam_tilt = [0.370972,  0.231609]
                mu_tilt = [0.184141,  0.196145]
            elif args.test_flow == 1:
                lam_tilt =  [0.277974,  0.402642]
                mu_tilt =  [0.366504,  0.329394]
    elif args.test_case == 3:
        lam = [0.2, 0.1]
        mu = [1.0, 1.0]
        if args.test_flow == 0:
            args.gamma_ref = 14.8766
        elif args.test_flow == 1:
            args.gamma_ref =  26.8068
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        elif args.CoE_phase == 1:
            lam_tilt =  [2/3, 1/3]
            mu_tilt = [0.3, 0.3]
        elif args.CoE_phase == 2:
            if args.test_flow == 0:
                lam_tilt = [0.611337,  0.132585]  
                mu_tilt = [0.234935,  0.234891 ]
            elif args.test_flow == 1:
                lam_tilt =  [0.417019,   0.17461 ]
                mu_tilt = [0.409197,  0.433389]
        

        
      
    for idx in range(args.unbiased_check):
        L = main(lam, mu, lam_tilt, mu_tilt, args.tile, random, args.batch_quantile_flag, idx)
        import csv 
        with open(f'./results/unbiased_check/case{args.test_case}_59_phase{args.CoE_phase}_K{num_type}_flow{args.test_flow}_n{args.n_cycles}_quantile.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerows([L])