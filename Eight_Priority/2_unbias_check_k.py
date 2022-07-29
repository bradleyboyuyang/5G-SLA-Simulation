import argparse
import csv
import time
import numpy as np

from distribution import ExponDistri
from simulation import static_priority_sim
from utils import cycle_statistics, generate_parameters, mean_CI, cycle_sta1


def get_flow_result(num_type, start_local, flow_id:int, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L:list, start, unbias_check, real, deno):
    """
    target: obtain typical statistics of the events with type "flow_id"
    theory: theoretical waiting time
    """
    L[flow_id].extend([args.n_cycles, time.time()-start_local, sum(nums)])
    cycle_num = [cycle_num[flow_id] for cycle_num in cycle_nums]
    cycle_sum = [cycle_sum[flow_id] for cycle_sum in cycle_sums]

    mean, halfCI = mean_CI(cycle_num)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
    print(f'Mean response time in flow {flow_id} (Denominator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    mean, halfCI = mean_CI(cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id} (Numerator): {mean:.14g} and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
    # mean, halfCI = cycle_statistics(cycle_num, cycle_sum)
    mean, halfCI = cycle_sta1(deno[flow_id][0], deno[flow_id][1], cycle_sum)
    L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
    print(f'Mean response time in flow {flow_id}: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g} (theoretic {theory:.14g} s)')
    print("=====================================================================================================================================")

    for i, gamma in enumerate(threshold[flow_id]):
        cycle_sum_prob = np.array([cycle_sum_prob[flow_id][i] for cycle_sum_prob in cycle_sum_probs])
        mean, halfCI = mean_CI(cycle_sum_prob)

        # 1.difference is smaller than half CI; 2. two CIs should overlap
        # unbias_check.extend([abs(mean-real[flow_id][i])<halfCI, mean-halfCI<real_upper[flow_id][i] and mean+halfCI>real_lower[flow_id][i]])

        L[flow_id].extend([gamma, mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96, time.time()-start])
        print(f'Prob (gamma={gamma}) in flow {flow_id} (Numerator): {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        
        # check whether cover the true probability value
        # mean, halfCI = cycle_statistics(cycle_num, cycle_sum_prob)
        mean, halfCI = cycle_sta1(deno[flow_id][0], deno[flow_id][1], cycle_sum_prob)

        unbias_check.extend([abs(mean-real[flow_id][i])<halfCI])

        L[flow_id].extend([mean, f'[{mean-halfCI:.4g},{mean+halfCI:.4g}]', halfCI, halfCI/mean/1.96])
        print(f'Prob (gamma={gamma}) in flow {flow_id}: {mean:.14g}  and CI [{mean-halfCI:.14g},{mean+halfCI:.14g}] RE {halfCI/1.96/mean:.14g}')
        freq = np.array([reason[flow_id][i] for reason in reasons])
        assert len(freq) == len(cycle_sum_prob)

        for k in range(num_type):
            print(f'Mean flow {k}: {np.mean(cycle_sum_prob[freq==k])}')
            print(f'Std flow {k}: {np.std(cycle_sum_prob[freq==k])}')
            print(f'ratio flow {k}: {sum(freq==k)/len(freq)}')
            L[flow_id].extend([sum(freq==k)/len(freq)]) # mean flow (ratio) of type k
            L[flow_id].extend([np.mean(cycle_sum_prob[freq==k]), np.var(cycle_sum_prob[freq==k])]) # likelihood, variance of type k event
            L[flow_id].extend([sum(freq==k)/(sum(freq!=-1))]) # conditional ratio of type k event
        L[flow_id].extend([sum(freq!=-1)/len(freq), np.var(cycle_sum_prob)]) # total ratio and total variance
        print("=====================================================================================================================================")
    return L, unbias_check

def main(idx, lam, mu, port_rate, threshold, args, mix_p, num_type, start, real, deno):
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
    rho = np.sum(rho_each)
    ES, ES2 = np.sum(p_each/mu_each), 2*np.sum(p_each/mu_each**2)
    Se = ES2 / ES / 2

    # simulation starts
    print("=====================================================================================================================================")
    print(f'({time.time()-start_local:.2f}s)Run simulation...')
    result = static_priority_sim(num_type, lam, mu, threshold, args.test_flow, port_rate, ExponDistri, args.cpu_nums, args.n_cycles, start_local, p=mix_p, policy=args.policy, record={'num_packet': True, 'cycle_len':True, 'response_sum':True, 'prob_sum':True, 'reason':True})
    cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums = result['cycle_len'], result['response_sum'], result['prob_sum'], result['reason'], result['num_packet']

    # obtain statistics
    print("=====================================================================================================================================")
    print(f'({time.time()-start_local:.2f}s)Statistics:')
    for flow_id in range(num_type):
        # calculate theoretical sojourn time (mean waiting time + service time)
        theory = rho*Se/(1-rho_each[:flow_id+1].sum())/(1-rho_each[:flow_id].sum()) + 1/mu_each[flow_id]
        # obtain simulation result for each type of events
        L, unbias_check = get_flow_result(num_type, start_local, flow_id, cycle_nums, cycle_sums, cycle_sum_probs, reasons, nums, theory, threshold, L, start, unbias_check, real, deno)
    print("total time: ", time.time()-start)
    print("summation of nums: ", sum(nums))
    print("unbias_check: ", unbias_check)
    print("=====================================================================================================================================")

    with open(f"result/2_unbias_check/{args.file_name}_t{args.test_flow}_iter{args.num_runs}_policy{args.policy}_K{num_type}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows([sum(L, [])])
    print(f'({time.time()-start_local:.2f}s) END')
    return unbias_check

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
    parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of the training dataset')
    parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
    parser.add_argument('-t', '--test_flow', type=int, default=4, help='which flow to switch measure')
    parser.add_argument('-k', '--num_runs', type=int, default=100, help='number of runs')
    parser.add_argument('-f', '--file_name', type=str, default='temp')
    parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
    args = parser.parse_args()

    lam = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    lam_tilt = [None, None, None, None, None, None, None, None]
    mu = [1, 1, 1, 1, 1, 1, 1, 1]
    mu_tilt = [None, None, None, None, None, None, None, None]
    lam, mu, num_type = generate_parameters(lam=lam, mu=mu, lam_tilt=lam_tilt, mu_tilt=mu_tilt)

    # # parameter for class 0
    # lam[:,1] = [0.189277519632802, 0.110356258346201, 0.120889598561375, 0.117341401283909, 0.119074930288464, 0.114216599922642, 0.120415804810095, 0.118167008971398]
    # mu[:,1] = [0.401716885892947, 0.711356399001901, 0.741132013921477, 0.566452649771021, 0.737662237409898, 0.589377568058318, 0.783039937263574, 0.494894212260086]

    # # parameter for class 1
    # lam[:,1] = [0.12962021001401, 0.147445452297473, 0.114811728799627, 0.114724868683136, 0.11266597351969, 0.117076068598556, 0.113217578534962, 0.115178767422004]
    # mu[:,1] = [0.679141367093629, 0.635126727642566, 0.839606850159693, 0.845382595231554, 0.850200449959887, 0.815793197700626, 0.79970173095683, 0.714628549026243]

    # # parameter for class 2
    # lam[:,1] = [0.117206088200824, 0.123581882602704, 0.138579572005877, 0.116510014364811, 0.112641024066951, 0.115519054546807, 0.111541102284489, 0.114319500310846]
    # mu[:,1] = [0.810173937500073, 0.76126064390643, 0.752583546661935, 0.848821855766813, 0.855577226662667, 0.848319973166271, 0.863606316153946, 0.794194595735813]

    # # parameter for class 3
    # lam[:,1] = [0.116592171252497, 0.122338680613305, 0.12080792511818, 0.142544982856913, 0.113114658376083, 0.11292817558195, 0.111097710044156, 0.114876434990032]
    # mu[:,1] = [0.824763891815725, 0.80239234644412, 0.826138891174883, 0.790122789507926, 0.841012773364193, 0.82576248307472, 0.840447956444283, 0.804906625776954]

    # parameter for class 4
    lam[:,1] = [0.120133205521678, 0.122994936174507, 0.118582412831034, 0.119255542692529, 0.138831612793011, 0.111902752843934, 0.113357609677306, 0.115123610739648]
    mu[:,1] = [0.81747505839341, 0.816224042951053, 0.80525952370193, 0.820337373962118, 0.787589256882753, 0.835404566149861, 0.825178155311421, 0.777268928997457]

    # # parameter for class 5
    # lam[:,1] = [0.122302809090588, 0.120086056545003, 0.118139026270029, 0.120793601942812, 0.116938401757487, 0.137481917657011, 0.111230034928763, 0.10970763054098]
    # mu[:,1] = [0.817364554107393, 0.806284220199324, 0.820879334830232, 0.805927426746803, 0.820957176724544, 0.809952517772643, 0.838044637208713, 0.779420029043509]

    # # parameter for class 6
    # lam[:,1] = [0.118569700264211, 0.123350308274175, 0.121251598074761, 0.118675680732426, 0.118342772069272, 0.124917070837515, 0.149065672409384, 0.115435642200199]
    # mu[:,1] = [0.794657216097302, 0.786588180796203, 0.805737806228643, 0.811295817752467, 0.805947311381577, 0.797117995637853, 0.794228743325187, 0.737099379554901]

    # # parameter for class 7
    # lam[:,1] = [0.119452987449119, 0.120782616057235, 0.124060185709526, 0.124765575471186, 0.120039646855997, 0.124224306854592, 0.123116150304057, 0.152636022314572]
    # mu[:,1] = [0.77432084447772, 0.782890701027899, 0.799933005755878, 0.78885528560104, 0.810507352574565, 0.779013292954761, 0.788777893305831, 0.781307786357912]

    ## thresholds (must be increasing for each flow)
    threshold = [[10] for _ in range(num_type)]
    real = [[0.000669478924356042435231952916781], [0.0036127871107343864840480020203595], [0.012649848515564681060522373800053], [0.033281871624842460508765292696539], [0.072418503692644969676263106873157], [0.13770964178876379356259140737605], [0.23679365581944631671850327533174], [0.37661889789252304101984230885306]]

    # initialization
    np.random.seed(42)
    mix_p = None
    start = time.time()
    port_rate = 1 * 8 # 1 byte per second

    deno = [[0.6297, 3.3744179100000005], [0.62715, 3.3533128775], [0.62393, 3.2987813551], [0.62455, 3.3740872975], [0.62925, 3.3333744375], [0.62203, 3.2661486791000005], [0.62654, 3.2955276284000004], [0.62223, 3.3446598271]]

    unbias_checks = [main(idx, lam, mu, port_rate, threshold, args, mix_p, num_type, start, real, deno) for idx in range(args.num_runs)]
    res = np.mean(unbias_checks, axis=0)
    print(res)
    with open(f"result/2_unbias_check/{args.file_name}_t{args.test_flow}_iter{args.num_runs}_policy{args.policy}_K{num_type}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([res])
