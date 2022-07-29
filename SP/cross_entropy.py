import time

import numpy as np
import pandas as pd
from ns.packet.distribution import ExponDistri
from simulation import static_priority_sim

from utils import cycle_statistics, sample_quantile


def CE_update(lam, mu, num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, rho, start, tile, gamma_ref=None):
    start = time.time()
    # simulate
    result = static_priority_sim(num_type, lam, mu, test_flow, gamma, port_rate, ExponDistri, n_cpu, n_cycles, start, True, {'max_response':True, 'save_response':True, 'cross_entropy':True})
    # get the quantile of max_response time
    gamma = np.quantile(result['max_response'], 1-rho)
    if gamma_ref is not None:
        gamma = min(gamma_ref, gamma)
    Q2_total = sample_quantile(result['response'], cycle_num, tile)
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
    return lam, mu, gamma, mean, halfCI, time.time()-start, Q2_total

def CE_update_quantile(lam, mu, num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, rho, start, tile, gamma_ref=None):
    start = time.time()
    CONTINUE = True
    # simulate
    result = static_priority_sim(num_type, lam, mu, test_flow, gamma, port_rate, ExponDistri, n_cpu, n_cycles, start, True, {'max_response':True, 'save_response':True, 'cross_entropy':True})
    # get the quantile of max_response time
    gamma = np.quantile(result['max_response'], 1-rho)
    if gamma_ref is not None:
        gamma = min(gamma_ref, gamma)
    Q2_total = sample_quantile(result['response'], cycle_num, tile)
    # compute the probability for the new gamma
    cycle_sum_prob = np.array([sum([(packet[0]>gamma)*np.exp(packet[1]) for packet in packet_cycle]) for packet_cycle in result['response']])
    mean, halfCI = cycle_statistics(cycle_num=cycle_num, cycle_sum=cycle_sum_prob)
    if mean + halfCI < 1 - tile:
        gamma = Q2_total
        cycle_sum_prob = np.array([sum([(packet[0]>gamma)*np.exp(packet[1]) for packet in packet_cycle]) for packet_cycle in result['response']])
        mean, halfCI = cycle_statistics(cycle_num=cycle_num, cycle_sum=cycle_sum_prob)
        CONTINUE = False
    CE_para = np.array(result['cross_entropy'])
    CE = CE_para * cycle_sum_prob.reshape(-1,1,1)
    CE = np.mean(CE, axis=0)
    assert CE.shape[0] == 4
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, gamma, mean, halfCI, time.time()-start, Q2_total, CONTINUE

def CE_stable(lam, mu, num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, tile):
    start = time.time()
    # simulate
    result = static_priority_sim(num_type, lam, mu, test_flow, gamma, port_rate, ExponDistri, n_cpu, n_cycles, start, True, {'prob_sum':True, 'save_response':True, 'num_packet':True, 'cross_entropy':True})
    cycle_sum_prob = np.array(result['prob_sum'])
    mean, halfCI = cycle_statistics(cycle_num=cycle_num, cycle_sum=cycle_sum_prob)
    Q2_total = sample_quantile(result['response'], cycle_num, tile)
    CE_para = np.array(result['cross_entropy'])
    CE = CE_para * cycle_sum_prob.reshape(-1,1,1)
    CE = np.mean(CE, axis=0)
    arrival_num, arrival_sum, service_num, service_sum = CE
    lam[:, 1] = arrival_num / arrival_sum
    mu[:, 1] = service_num / service_sum
    return lam, mu, mean, halfCI, time.time()-start, Q2_total

def CE_main(num_type, lam_ref, mu_ref, cycle_num, port_rate, rho, test_flow, n_cpu, n_cycles, tile, num_runs, test_cycles, file_name, gamma_ref=None):
    start_local = time.time()
    print(f'(+{time.time()-start_local:.2f}s)Run Cross-Entropy Method...')
    # initial change of measure
    initial_phase = 0
    lam = lam_ref[:, [0, initial_phase]].copy()
    mu = mu_ref[:, [0, initial_phase]].copy()

    infos = []
    CONTINUE = True 
    gamma_u, gamma_l, gamma = 0, 0, 0
    print(f'(+{time.time()-start_local:.2f}s)Update and increase the gamma using {n_cycles} cycles...')
    while CONTINUE:
        info = list(lam[:,1]) + list(mu[:,1])
        # update parameters
        lam, mu, gamma, mean, halfCI, time_iter, Q2_total = CE_update(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, rho, start_local, tile, gamma_ref=gamma_ref)
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
        info.extend([mean, RE, Q2_total, time_iter])
        infos.append(info)
    print(f'(+{time.time()-start_local:.2f}s)Fix gamma and update, to stabilize the parameter, using {n_cycles} cycles...')
    for _ in range(num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, mean, halfCI, time_iter, Q2_total = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, tile)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, Q2_total, time_iter])
        infos.append(info)
    # save last one
    info = list(lam[:,1]) + list(mu[:,1])
    info.append(gamma)
    info.extend([None, None, None, None])
    infos.append(info)
    print(f'(+{time.time()-start_local:.2f}s)End the cross-entropy simulation:')
    df = pd.DataFrame(np.array(infos), columns=[f'lam{k+1}' for k in range(num_type)]+[f'mu{k+1}' for k in range(num_type)]+['gamma', 'prob', 'RE', 'quantile', 'time'])
    # print(df)
    df.to_csv(f'./results/cross_entropy/{file_name}_t{test_flow}_tile{tile}_r{rho}_K{num_type}_gamma{gamma}_cecycle{n_cycles}.csv')
    print('where the last line is the final result.')

    return lam, mu, gamma_l, gamma_u, gamma

def CE_up_down(num_type, lam_ref, mu_ref, cycle_num, port_rate, rho, test_flow, n_cpu, n_cycles, tile, num_runs, test_cycles, file_name, gamma_ref=None):
    assert gamma_ref is None
    start_local = time.time()
    print(f'(+{time.time()-start_local:.2f}s)Run Cross-Entropy Method...')
    # initial change of measure
    initial_phase = 0
    lam = lam_ref[:, [0, initial_phase]].copy()
    mu = mu_ref[:, [0, initial_phase]].copy()

    infos = []
    CONTINUE = True 
    gamma_u, gamma_l, gamma = 0, 0, 0
    print(f'(+{time.time()-start_local:.2f}s)Update and increase the gamma using {n_cycles} cycles...')
    while CONTINUE:
        info = list(lam[:,1]) + list(mu[:,1])
        # update parameters
        lam, mu, gamma, mean, halfCI, time_iter, Q2_total = CE_update(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, rho, start_local, tile, gamma_ref=gamma_ref)
        info.append(gamma)
        if mean + halfCI < 1 - tile:
            gamma_u = gamma
            CONTINUE = False
        elif mean - halfCI > 1 - tile:
            gamma_l = gamma
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, Q2_total, time_iter])
        infos.append(info)
        
    print(f'(+{time.time()-start_local:.2f}s)Fix gamma and update, to stabilize the parameter, using {n_cycles} cycles...')
    for _ in range(1):
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, mean, halfCI, time_iter, Q2_total = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, tile)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, Q2_total, time_iter])
        infos.append(info)
    print(f'(+{time.time()-start_local:.2f}s)Use quantile as gamma and update, to stabilize the parameter, using {n_cycles} cycles...')
    gamma = Q2_total
    for _ in range(num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, mean, halfCI, time_iter, Q2_total = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, tile)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, Q2_total, time_iter])
        infos.append(info)
    # save last one
    info = list(lam[:,1]) + list(mu[:,1])
    info.append(gamma)
    info.extend([None, None, None, None])
    infos.append(info)
    print(f'(+{time.time()-start_local:.2f}s)End the cross-entropy simulation:')
    df = pd.DataFrame(np.array(infos), columns=[f'lam{k+1}' for k in range(num_type)]+[f'mu{k+1}' for k in range(num_type)]+['gamma', 'prob', 'RE', 'quantile', 'time'])
    # print(df)
    df.to_csv(f'./results/cross_entropy/{file_name}_t{test_flow}_tile{tile}_r{rho}_K{num_type}_gamma{gamma}_cecycle{n_cycles}.csv')
    print('where the last line is the final result.')

    return lam, mu, gamma_l, gamma_u, gamma

def CE_up(num_type, lam_ref, mu_ref, cycle_num, port_rate, rho, test_flow, n_cpu, n_cycles, tile, num_runs, test_cycles, file_name, gamma_ref=None):
    assert gamma_ref is None
    start_local = time.time()
    print(f'(+{time.time()-start_local:.2f}s)Run Cross-Entropy Method...')
    # initial change of measure
    initial_phase = 0
    lam = lam_ref[:, [0, initial_phase]].copy()
    mu = mu_ref[:, [0, initial_phase]].copy()

    infos = []
    CONTINUE = True 
    gamma_u, gamma_l, gamma = 0, 0, 0
    print(f'(+{time.time()-start_local:.2f}s)Update and increase the gamma using {n_cycles} cycles...')
    while CONTINUE:
        info = list(lam[:,1]) + list(mu[:,1])
        # update parameters
        lam, mu, gamma, mean, halfCI, time_iter, Q2_total, CONTINUE = CE_update_quantile(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, rho, start_local, tile, gamma_ref=gamma_ref)
        info.append(gamma)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, Q2_total, time_iter])
        infos.append(info)
        
    print(f'(+{time.time()-start_local:.2f}s)Fix gamma and update, to stabilize the parameter, using {n_cycles} cycles...')
    for _ in range(num_runs):
        info = list(lam[:,1]) + list(mu[:,1])
        info.append(gamma)
        lam, mu, mean, halfCI, time_iter, Q2_total = CE_stable(lam.copy(), mu.copy(), num_type, port_rate, gamma, test_flow, cycle_num, n_cpu, n_cycles, tile)
        RE = halfCI / 1.96 / mean
        info.extend([mean, RE, Q2_total, time_iter])
        infos.append(info)
    # save last one
    info = list(lam[:,1]) + list(mu[:,1])
    info.append(gamma)
    info.extend([None, None, None, None])
    infos.append(info)
    print(f'(+{time.time()-start_local:.2f}s)End the cross-entropy simulation:')
    df = pd.DataFrame(np.array(infos), columns=[f'lam{k+1}' for k in range(num_type)]+[f'mu{k+1}' for k in range(num_type)]+['gamma', 'prob', 'RE', 'quantile', 'time'])
    # print(df)
    df.to_csv(f'./results/cross_entropy/{file_name}_t{test_flow}_tile{tile}_r{rho}_K{num_type}_gamma{gamma}_cecycle{n_cycles}.csv')
    print('where the last line is the final result.')

    return lam, mu, gamma_l, gamma_u, gamma