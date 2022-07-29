import time

import numpy as np
import pandas as pd
from ns.packet.distribution import ExponDistri

from simulation import static_priority_sim


def CE_stable(lam, mu, num_type, port_rate, gamma, test_flow, n_cpu, n_cycles):
    start = time.time()
    # simulate
    result = static_priority_sim(num_type, lam, mu, test_flow, np.inf, port_rate, ExponDistri, n_cpu, n_cycles, start, True, {'cycle_len':True, 'cross_entropy':True})
    cycle_sum = np.array(result['cycle_len'])
    mean, std = np.mean(cycle_sum), np.std(cycle_sum)/np.sqrt(len(cycle_sum))
    # print(mean, std, std/mean)
    CE_para = np.array(result['cross_entropy'])
    CE = CE_para * cycle_sum.reshape(-1,1,1)
    CE = np.mean(CE, axis=0)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, mean, std, time.time()-start

def CE_nominator(num_type, lam_ref, mu_ref, port_rate, rho, test_flow, n_cpu, n_cycles, num_runs, file_name):
    start_local = time.time()
    print(f'(+{time.time()-start_local:.2f}s)Run Cross-Entropy Method...')
    # initial change of measure
    initial_phase = 0
    lam = lam_ref[:, [0, initial_phase]].copy()
    mu = mu_ref[:, [0, initial_phase]].copy()

    infos = []
    print(f'(+{time.time()-start_local:.2f}s)Update and increase the gamma using {n_cycles} cycles...')
    for _ in range(num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        # info.append(gamma)
        lam, mu, mean, std, time_iter = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, np.inf, test_flow, n_cpu, n_cycles)
        RE = std / mean
        info.extend([mean, RE, time_iter])
        infos.append(info)
        
    # save last one
    info = list(lam[:,1]) + list(mu[:,1])
    # info.append(gamma)
    info.extend([None, None, None])
    infos.append(info)
    print(f'(+{time.time()-start_local:.2f}s)End the cross-entropy simulation:')
    df = pd.DataFrame(np.array(infos), columns=[f'lam{k+1}' for k in range(num_type)]+[f'mu{k+1}' for k in range(num_type)]+['mean', 'RE', 'time'])
    print(df)
    df.to_csv(f'./results/cross_entropy/{file_name}_t{test_flow}_r{rho}_K{num_type}_nominator_cecycle{n_cycles}.csv')
    print('where the last line is the final result.')

    return lam, mu