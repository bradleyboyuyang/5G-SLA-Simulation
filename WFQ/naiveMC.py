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
parser.add_argument('--tile', type=float, default=0.999)
parser.add_argument('--gamma_ref', type=float, default=None)
parser.add_argument('--gamma_p', type=float, default=0.001)
parser.add_argument('--batch_quantile_flag', type=int, default=None)
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
    
    result = static_priority_sim(num_type, arrival_rates, service_rates, args.test_flow, args.gamma_ref, 8, ExponDistri, args.cpu_nums, args.n_cycles, time.time(), True, {"prob_sum":True, "cycle_len":True,'save_response':True, 'reason': True})
    cycle_len = result['cycle_len']
    mean, std = np.mean(cycle_len), np.std(cycle_len)/np.sqrt(len(cycle_len))
    print(f'mean {mean}, std {std}, RE {std/mean}')
    L.extend([len(cycle_len), mean, std, std/mean])

    nominator = result['prob_sum']
    mean, std = np.mean(nominator), np.std(nominator)/np.sqrt(len(nominator))
    print(f'{args.gamma_ref}: mean {mean}, std {std}, RE {std/mean}')
    L.extend([args.gamma_ref, len(nominator), mean, std, std/mean])
    mean, CI = cycle_statistics(cycle_num=cycle_len, cycle_sum=nominator)
    print(f'{args.gamma_ref}: mean {mean}, CI ({mean-CI}, {mean+CI}), RE {CI/1.96/mean}')
    L.extend([args.gamma_ref, mean, mean-CI, mean+CI, CI/1.96/mean])

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
    random = np.random.RandomState(42)
    # num_type = 4
    # lam = [0.15, 0.15, 0.15, 0.15]
    # mu = [1, 1, 1, 1]
    # lam_tilt = [0.3918284534863452, 0.1917008884432369, 0.19384927003983013, 0.19015014095684432]
    # mu_tilt = [0.30289029745149215, 0.4324399899548464, 0.39608747332020994, 0.34511310915205345]
    
    num_type = 2
    if args.test_case == 0:
        lam = [0.1, 0.2]
        mu = [0.5, 0.5]
        if args.test_flow == 0:
            args.gamma_ref = 19.774
        elif args.test_flow == 1:
            args.gamma_ref =  47.448
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        elif args.CoE_phase == 1:
            lam_tilt = [0.2063278712020342,	0.33333333333333326]
            mu_tilt = [0.24233274791576997,	0.3]
        elif args.CoE_phase == 2:
            if args.test_flow == 0:
                lam_tilt = [0.238458,  0.266307]  
                mu_tilt = [0.157736,  0.197682]
            elif args.test_flow == 1:
                lam_tilt =  [0.170515,  0.357432]  
                mu_tilt =  [0.271787,  0.278711 ]
    elif args.test_case == 1:
        lam = [0.2, 0.1]
        mu = [0.5, 0.5]
        if args.test_flow == 0:
            args.gamma_ref = 24.368
        elif args.test_flow == 1:
            args.gamma_ref =  74.1107
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        elif args.CoE_phase == 1:
            lam_tilt = [0.33333333333333326, 0.16666666666666663]
            mu_tilt = [0.3, 0.3]
        elif args.CoE_phase == 2:
            if args.test_flow == 0:
                lam_tilt =  [0.405868,  0.116172]
                mu_tilt =  [0.202627,  0.190995]
            elif args.test_flow == 1:
                lam_tilt =  [0.317101,  0.158033  ]
                mu_tilt =  [0.304031,  0.312383]
    elif args.test_case == 2:
        lam = [0.1, 0.2]
        mu = [1.0, 1.0]
        if args.test_flow == 0:
            args.gamma_ref = 8.524
        elif args.test_flow == 1:
            args.gamma_ref =  11.5409
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        elif args.CoE_phase == 1:
            lam_tilt =  [1/3, 2/3]
            mu_tilt = [0.3, 0.3]
        elif args.CoE_phase == 2:
            if args.test_flow == 0:
                lam_tilt = [ 0.307337,  0.238571]
                mu_tilt = [0.20956,  0.252305]
            elif args.test_flow == 1:
                lam_tilt =  [0.218741,  0.434699]
                mu_tilt =  [0.341315,  0.338675]
    elif args.test_case == 3:
        lam = [0.2, 0.1]
        mu = [1.0, 1.0]
        if args.test_flow == 0:
            args.gamma_ref = 9.0717
        elif args.test_flow == 1:
            args.gamma_ref =  13.9981
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        elif args.CoE_phase == 1:
            lam_tilt =  [2/3, 1/3]
            mu_tilt = [0.3, 0.3]
        elif args.CoE_phase == 2:
            if args.test_flow == 0:
                lam_tilt = [0.458901,  0.120053]
                mu_tilt = [0.243433,  0.233283]
            elif args.test_flow == 1:
                lam_tilt =  [0.385957,  0.229724]
                mu_tilt =  [0.404358,  0.448357]
    elif args.test_case == 4:
        lam = [0.1, 0.2]
        mu = [0.6, 0.6]
        if args.test_flow == 0:
            args.gamma_ref =  15.7145
        elif args.test_flow == 1:
            args.gamma_ref =  30.3134 
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        elif args.CoE_phase == 1:
            lam_tilt =  [0.23744432803426394, 0.4 ]
            mu_tilt = [0.2526908117651133, 0.3 ]
        elif args.CoE_phase == 2:
            if args.test_flow == 0:
                lam_tilt = [0.250314,  0.254367 ]
                mu_tilt = [0.158534,  0.201804 ]
            elif args.test_flow == 1:
                lam_tilt =  [0.207677,  0.379051]
                mu_tilt =  [0.294608,  0.321686]
    elif args.test_case == 5:
        lam = [0.2, 0.4]
        mu = [2, 1]
        if args.test_flow == 0:
            args.gamma_ref =  6.925240475373265
        elif args.test_flow == 1:
            args.gamma_ref =  16.679497969069484
        if args.CoE_phase == 0:
            lam_tilt = lam.copy()
            mu_tilt = mu.copy()
        

        
      
    for idx in range(args.unbiased_check):
        L = main(lam, mu, lam_tilt, mu_tilt, args.tile, random, args.batch_quantile_flag, idx)
        import csv 
        with open(f'./results/unbiased_check/case{args.test_case}_phase{args.CoE_phase}_K{num_type}_flow{args.test_flow}_n{args.n_cycles}_quantile.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerows([L])