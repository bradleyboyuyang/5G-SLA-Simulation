import argparse
import csv
import datetime
import time

import numpy as np

from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_statistics, generate_parameters, mean_CI


def get_flow_result(num_type, start_local, flow_id:int, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L:list, start):
    """
    target: obtain typical statistics of the events with type "flow_id"
    theory: theoretical response time
    """
    L[flow_id].extend([args.n_cycles, time.time()-start_local, sum(nums)])
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]

    # statistics of sojourn time
    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow {flow_id+1} (Denominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id+1} (Numerator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id+1}: {mean:.14g}s  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {theory:.14g}s)')
    print("=====================================================================================================================================")

    # statistics of probability
    for i, gamma in enumerate(threshold[flow_id]):
        cycle_sum_prob = np.array([cycle_sum_prob[flow_id][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        L[flow_id].extend([gamma, mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob (gamma={gamma}) in flow {flow_id+1} (Numerator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob (gamma={gamma}) in flow {flow_id+1}: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[flow_id][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)

        # supplementary statistics
        for k in range(num_type):
            print(f'Mean flow {k+1}: {np.mean(cycle_sum_prob[freq==k])}')
            print(f'Std flow {k+1}: {np.std(cycle_sum_prob[freq==k])}')
            print(f'ratio flow {k+1}: {sum(freq==k)/len(freq)}')
            L[flow_id].extend([sum(freq==k)/len(freq)]) # mean flow (ratio) of type k
            L[flow_id].extend([np.mean(cycle_sum_prob[freq==k]), np.var(cycle_sum_prob[freq==k])]) # likelihood, variance of type k event
            L[flow_id].extend([sum(freq==k)/(sum(freq!=-1))]) # conditional ratio of type k event
        L[flow_id].extend([sum(freq!=-1)/len(freq), np.var(cycle_sum_prob)]) # total ratio and total variance
        print("=====================================================================================================================================")
    return L

def main(lam, mu, port_rate, threshold, args, mix_p, num_type, start):
    start_local = time.time()
    L = [[] for i in range(num_type)]
    L[0].append(args.idx)
    if mix_p is not None:
        L[0].extend(list(mix_p.keys()) + list(mix_p.values()))
    for i in range(num_type):
        L[i].extend(list(lam[i])+list(mu[i]))
    
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
    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, args.n_cycles, start, p=mix_p, policy=args.policy, record={'num_packet': True, 'cycle_len':True, 'response_sum':True, 'prob_sum':True, 'reason':True})
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums = result['cycle_len'], result['response_sum'], result['prob_sum'], result['reason'], result['num_packet']

    # obtain statistics
    print("=====================================================================================================================================")
    print(f'({time.time()-start_local:.2f}s)Statistics:')
    for flow_id in range(num_type):
        # calculate theoretical sojourn time (mean waiting time + service time)
        theory = rho*Se/(1-rho_each[:flow_id+1].sum())/(1-rho_each[:flow_id].sum()) + 1/mu_each[flow_id]
        # obtain simulation result for each type of events
        L = get_flow_result(num_type, start_local, flow_id, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L, start)
    print("total time: ", time.time()-start)
    print("number of packets: ", sum(nums))
    print("=====================================================================================================================================")

    # append the simulation record into a csv file
    # with open(f"result/1_run_para_{num_type}.csv", "a", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows([sum(L, [])])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
    parser.add_argument('--test_flow', type=int, default=0, help='which flow to switch measure')
    parser.add_argument('--idx', type=int, default=None, help='index of the instance')
    parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
    args = parser.parse_args()

    lam = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    mu = [1, 1, 1, 1, 1, 1, 1, 1]
    lam_tilt = [0.166649119026982, 0.109815680997942, 0.123933196553376, 0.109657015155136, 0.121423887174142, 0.120216863006439, 0.106059783076043, 0.123662379153521]
    mu_tilt = [0.41524511512576, 0.783508162559042, 0.699197133065559, 0.753981095619079, 0.639204961119899, 0.759638809056957, 0.737816039813978, 0.539858579004845]

    # initialization
    np.random.seed(42)
    mix_p = None
    lam, mu, num_type = generate_parameters(lam=lam, mu=mu, lam_tilt=lam_tilt, mu_tilt=mu_tilt)
    threshold = [[10] for _ in range(num_type)]

    start = time.time()
    port_rate = 1 * 8 # 1 byte per second
    if args.idx is None:
        args.idx = f'{datetime.datetime.now()}'
    main(lam, mu, port_rate, threshold, args, mix_p, num_type, start) 

