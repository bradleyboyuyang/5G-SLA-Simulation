import time
import numpy as np
from A_run_cycle_length import naive_cycle_length
from B_cross_entropy import CE_main
from C_evaluation import evaluate, evaluate2
import argparse

parser = argparse.ArgumentParser(description='Hyperparameters of Problem and NN')
parser.add_argument('-n', '--n_cycles', type=int, default=100000, help='the size of CE')
parser.add_argument('--rough_cycles', type=int, default=10000, help='the size of rough search')
parser.add_argument('-N','--test_cycles', type=int, default=100000, help='the size of the evaluation and precise')
parser.add_argument('--naive_cycles', type=int, default=1000000, help='the size of the naive simulation')
parser.add_argument('--cpu_nums', type=int, default=16, help='the number of cpus')
parser.add_argument('--precision', type=int, default=100, help='precision')
parser.add_argument('--tol', type=float, default=1, help='tolerance')
parser.add_argument('-t','--test_flow', type=int, default=1, help='which flow to switch measure')
parser.add_argument('--tile', type=float, default=0.999, help='threshold')
parser.add_argument('-k', '--num_runs', type=int, default=5, help='number of runs')
parser.add_argument('--rho', type=float, default=0.1, help='para of CE')
parser.add_argument('-f','--file_name', type=str, default='temp')
parser.add_argument('--policy', type=int, default=0, help='policy of switching measure')
parser.add_argument('--gamma_ref', type=float, default=12.281860113143921)
parser.add_argument('--unbiased_check', type=int, default=100)
args = parser.parse_args()

def tile_range(res, tile):
    lower = res[:,3]
    upper = res[:,4]
    l = res[:,0][lower > 1-tile]
    lower_tile = l[-1] if len(l) > 0 else res[0,0]
    u = res[:,0][upper < 1-tile]
    upper_tile = u[0] if len(u) > 0 else res[-1,0]
    return lower_tile, upper_tile

def mid_point(res, tile):
    l = res[:,0][res[:,1] > 1-tile]
    lower_tile = l[-1] if len(l) > 0 else res[0,0]
    u = res[:,0][res[:,1] < 1-tile]
    upper_tile = u[0] if len(u) > 0 else res[-1,0]
    return (lower_tile + upper_tile) / 2


if __name__ == '__main__':
    np.random.seed(42)
    start = time.time()

    num_type = 8
    lam = np.array([[0.1],[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]], dtype=float)
    mu = np.array([[1],[1], [1], [1], [1], [1], [1], [1]], dtype=float)

    theory_gamma = [9.528393745422363, 12.281860113143921, 16.625237464904785, 23.275141716003418, 33.947336077690125, 52.522705495357513, 89.005772769451141, 176.2980234622955]

    # real = [[0.000669478924356042435231952916781], [0.0036127871107343864840480020203595], [0.012649848515564681060522373800053], [0.033281871624842460508765292696539], [0.072418503692644969676263106873157], [0.13770964178876379356259140737605], [0.23679365581944631671850327533174], [0.37661889789252304101984230885306]]


    port_rate = 1 * 8 # 1 byte per second
    L = []
    print(f"***********************cycle length ({time.time()-start:.2f}s)********************************")
    cycle_length = naive_cycle_length(lam, mu, port_rate, args.test_cycles, 16, num_type)
    print(np.array(cycle_length))
    L.extend(list(lam.reshape(-1))+list(mu.reshape(-1))+[args.test_flow, args.test_cycles]+list(cycle_length[args.test_flow]))
    print(f"**********************cross entropy ({time.time()-start:.2f}s)********************************")
    lam, mu, gamma_l, gamma_u = CE_main(num_type, lam, mu, cycle_length, port_rate, args.rho, args.test_flow, args.cpu_nums, args.n_cycles, args.policy, args.tile, args.num_runs, None, args.file_name, args.gamma_ref)
    print(f"lambda {list(lam[:,1])} mu {list(mu[:,1])}")
    L.extend(list(lam[:,1].reshape(-1))+list(mu[:,1].reshape(-1))+[gamma_l,gamma_u,args.rho,args.n_cycles,args.policy,args.tile])
    if args.gamma_ref is None:
        print(f"**********************{args.tile}-tile ({time.time()-start:.2f}s)********************************")
        n_cycles = args.rough_cycles
        print(f"({time.time()-start:.2f}s) roughly narrow the range")
        gamma_range = -1
        while gamma_u - gamma_l > args.tol and not np.isclose(gamma_u-gamma_l, gamma_range):
        # for i in range(2):
            gamma_range = gamma_u - gamma_l
            thresholds = [[10] for _ in range(num_type)]
            thresholds[args.test_flow] = list(np.linspace(gamma_l, gamma_u, args.precision))
            res, lam, mu = evaluate(num_type, lam, mu, port_rate, thresholds, args.test_flow, args.cpu_nums, n_cycles, cycle_length[args.test_flow], None, args.policy, mute=True)
            gamma_l, gamma_u = tile_range(res, args.tile)
            print(f'gamma_range: ({gamma_l:.6f}, {gamma_u:.6f}) {gamma_u - gamma_l:.6f} interval {thresholds[args.test_flow][1] - thresholds[args.test_flow][0]}')
            n_cycles *= 1
        L.extend([args.tol, args.precision, args.test_flow])
        print(f"({time.time()-start:.2f}s) precisely narrow the range")
        thresholds = [[10] for _ in range(num_type)]
        thresholds[args.test_flow] = list(np.linspace(gamma_l, gamma_u, args.precision))
        res, lam, mu = evaluate(num_type, lam, mu, port_rate, thresholds, args.test_flow, args.cpu_nums, args.test_cycles, cycle_length[args.test_flow], None, args.policy)
        gamma_l, gamma_u = tile_range(res, args.tile)
        print(f'gamma_range: ({gamma_l:.6f}, {gamma_u:.6f}) {gamma_u - gamma_l:.6f} interval {thresholds[args.test_flow][1] - thresholds[args.test_flow][0]}')
        L.extend([gamma_l, gamma_u,thresholds[args.test_flow][1] - thresholds[args.test_flow][0]])
    print(f"**********************evaluate ({time.time()-start:.2f}s)********************************")
    for idx in range(args.unbiased_check):
        thresholds = [[10] for _ in range(num_type)]
        if args.gamma_ref is None:
            gamma = mid_point(res, args.tile)
            thresholds[args.test_flow] = [gamma]
        else:
            thresholds[args.test_flow] = [args.gamma_ref]
        print(lam, mu)
        # res, _, _ = evaluate(num_type, lam, mu, port_rate, thresholds, args.test_flow, args.cpu_nums, args.test_cycles, cycle_length[args.test_flow], None, args.policy)

        res, _, _ = evaluate2(idx, num_type, lam, mu, port_rate, thresholds, args.test_flow, args.cpu_nums, args.test_cycles, cycle_length[args.test_flow], None, args.policy, args, start)
        print(res)

        

        L.extend(list(res[0]) + [time.time()-start])
    # if args.naive_cycles is not None:
    #     print(f"************************Naive simulation ({time.time()-start:.2f}s)****************************")
    #     res = evaluate(num_type, lam[:,[0,0]], mu[:,[0,0]], port_rate, thresholds, args.test_flow, args.cpu_nums, args.naive_cycles, cycle_length[args.test_flow], None, args.policy)
    #     print(res[0])
    print(f"********************************END ({time.time()-start:.2f}s)********************************")
    import csv 
    with open(f'./result/4_q_tile_K{num_type}.csv', "a") as f:
        writer = csv.writer(f)
        writer.writerows([L])