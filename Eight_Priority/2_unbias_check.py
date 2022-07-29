import argparse
import csv
import time

import numpy as np

from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_statistics, mean_CI

parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of the training dataset')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('-t', '--test_flow', type=int, default=1, help='which flow to swtich measure')
parser.add_argument('-k', '--num_runs', type=int, default=500, help='number of runs')
parser.add_argument('-f', '--file_name', type=str, default='temp')
parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
args = parser.parse_args()

# parameters
mix_p = None
num_type = 2
## lam mu   original       1
lam = np.array([[0.6, None],
                [0.2, None]])
mu = np.array([[   2, None],
               [   1, None]])
# Thesis
lam[:,1] = [1.5,0.5]
mu[:,1] = [0.8, 0.4]
# ### Priority 5
# lam[:,1] = [0.8998870926555549,0.6142637224394023]
# mu[:,1] = [1.0758906952016303,0.2424444477171096]
### Ordinary 10 policy 0
# lam[:,1] = [0.8204897498205567,0.5791510463854309]
# mu[:,1] = [1.3366737345090036,0.4373962955149862]
### Ordinary 10 policy 1
# lam[:,1] = [0.8017380259860238,0.579801196549211]
# mu[:,1] = [1.3665636797944918,0.4237711449316795]
### Ordinary 10 policy 2
# lam[:,1] = [0.8394579259348436,0.5982263254131608]
# mu[:,1] = [1.3421835386475596,0.4413138790146214]
## thresholds (must be increasing for each flow)
threshold = [[5], [5, 10]]
real = [[0.0050488825], [0.025138833333333, 0.0038560725]]
real_lower = [[0.0050349732904737], [0.024898493401137,0.0038443439681686]]
real_upper = [[0.0050627917095263], [0.025379173265529,0.0038678010318314]]

# initial
np.random.seed(42)
start = time.time()
port_rate = 1 * 8 # 1 byte per second

def main(idx, lam, mu, port_rate, threshold, args, mix_p, num_type):
    unbias_check = []

    start_local = time.time()
    L = [[] for i in range(num_type)]
    L[0].append(idx)
    if mix_p is not None:
        L[0].extend(list(mix_p.keys()) + list(mix_p.values()))
    for i in range(num_type):
        L[i].extend(list(lam[i])+list(mu[i]))
    
    
    lam_each = lam[:,0]
    mu_each = mu[:,0]
    p = lam / np.sum(lam, axis=0, keepdims=True)
    p_each = p[:,0]
    rho_each = lam_each / mu_each
    rho_total = np.sum(rho_each)
    lam1, lam2 = lam_each
    mu1, mu2 = mu_each
    rho = rho_total
    rho1, rho2 = rho_each

    print(f'({time.time()-start_local:.2f}s)Run simulation...')
    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, args.n_cycles, start_local, p=mix_p, policy=args.policy, record={'num_packet': True, 'cycle_len':True, 'response_sum':True, 'prob_sum':True, 'reason':True})
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums = result['cycle_len'], result['response_sum'], result['prob_sum'], result['reason'], result['num_packet']

    print(f'({time.time()-start_local:.2f}s)Statistics:')

    flow_id = 0
    L[flow_id].extend([args.n_cycles, time.time()-start_local, sum(nums)])
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]
            
    ES, ES2 = np.sum(p_each/mu_each), 2*np.sum(p_each/mu_each**2)
    Se = ES2 / ES / 2
    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow 1 (Denominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow 1 (Nominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow 1: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {rho*Se/(1-rho1)+1/mu1:.14g} s)')

    for i, gamma in enumerate(threshold[flow_id]):
        cycle_sum_prob = np.array([cycle_sum_prob[flow_id][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        unbias_check.extend([abs(mean-real[flow_id][i])<halfCI, mean-halfCI<real_upper[flow_id][i] and mean+halfCI>real_lower[flow_id][i]])
        L[flow_id].extend([gamma, mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob {gamma} in flow 1 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob {gamma} in flow 1: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[0][i] for reason in reasons])
        print(f'Mean flow 1 {np.mean(cycle_sum_prob[freq==0])}')
        print(f'ratio flow 1 {sum(freq==0)/len(freq)}')
        print(f'Mean flow 2 {np.mean(cycle_sum_prob[freq==1])}')
        print(f'ratio flow 2 {sum(freq==1)/len(freq)}')
        L[flow_id].extend([sum(freq==0)/len(freq), sum(freq==1)/len(freq), sum(freq==0)/len(freq) + sum(freq==1)/len(freq), sum(freq==1)/(sum(freq==0)+sum(freq==1))])
        L[flow_id].extend([np.mean(cycle_sum_prob[freq==0]), np.mean(cycle_sum_prob[freq==1])])
        L[flow_id].extend([np.var(cycle_sum_prob[freq==0]), np.var(cycle_sum_prob[freq==1])])
        L[flow_id].append(np.var(cycle_sum_prob))


    flow_id = 1
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]
    # cycle_sum_prob = packets_info[:, 5]
    # reasons = [info[3][1] for info in packets_info]    
    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow 2 (Denominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow 2 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow 2: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {rho*Se/(1-rho1)/(1-rho)+1/mu2:.14g} s) ')
    for i, gamma in enumerate(threshold[1]):
        cycle_sum_prob = np.array([cycle_sum_prob[1][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)
        unbias_check.extend([abs(mean-real[flow_id][i])<halfCI, mean-halfCI<real_upper[flow_id][i] and mean+halfCI>real_lower[flow_id][i]])
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob {gamma} in flow 2 (Nominator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        print(f'std {np.std(cycle_sum_prob)}')
        mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob {gamma} in flow 2: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[1][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)
        print(f'Mean flow 1 {np.mean(cycle_sum_prob[freq==0])}')
        print(f'Std flow 1 {np.std(cycle_sum_prob[freq==0])}')
        print(f'ratio flow 1 {sum(freq==0)/len(freq)}')
        print(f'Mean flow 2 {np.mean(cycle_sum_prob[freq==1])}')
        print(f'Std flow 2 {np.std(cycle_sum_prob[freq==1])}')
        print(f'ratio flow 2 {sum(freq==1)/len(freq)}')
        L[flow_id].extend([sum(freq==0)/len(freq), sum(freq==1)/len(freq), sum(freq==0)/len(freq) + sum(freq==1)/len(freq), sum(freq==1)/(sum(freq==0)+sum(freq==1))])
        L[flow_id].extend([np.mean(cycle_sum_prob[freq==0]), np.mean(cycle_sum_prob[freq==1])])
        L[flow_id].extend([np.var(cycle_sum_prob[freq==0]), np.var(cycle_sum_prob[freq==1])])
        L[flow_id].append(np.var(cycle_sum_prob))
    print(time.time()-start)
    print(sum(nums))
    print(unbias_check)

    with open(f"result/2_unbias_check/{args.file_name}_t{args.test_flow}_iter{args.num_runs}_policy{args.policy}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerows([sum(L, [])])
    print(f'({time.time()-start_local:.2f}s) END')
    return unbias_check

    

if __name__ == '__main__':
    unbias_checks = [main(idx, lam, mu, port_rate, threshold, args, mix_p, num_type) for idx in range(args.num_runs)]
    res = np.mean(unbias_checks, axis=0)
    print(res)
    with open(f"result/2_unbias_check/{args.file_name}_t{args.test_flow}_iter{args.num_runs}_policy{args.policy}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerows([res])
