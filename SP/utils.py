import itertools

import numpy as np


def cycle_statistics(cycle_num, cycle_sum):
    m2 = len(cycle_num)
    m1 = len(cycle_sum)
    R_hat = np.mean(cycle_sum)
    tau_hat = np.mean(cycle_num)
    l_hat = R_hat / tau_hat
    S11 = np.var(cycle_sum)
    S22 = np.var(cycle_num)
    S2 = S11 + l_hat**2 * S22 
    half_CI = 1.96 * np.sqrt(S2 / min(m1,m2)) / tau_hat
    return l_hat, half_CI

def sample_quantile(result, cycle, p):
    m1 = len(result)
    m2 = len(cycle)
    beta_n = sum(cycle)
    response = sorted(itertools.chain.from_iterable(result), reverse=True)
    response = np.array(response)
    H, L = response[:,0], np.exp(response[:,1])
    L_cumsum = np.cumsum(L)
    idx = np.searchsorted(L_cumsum, beta_n*(1-p)/m2*m1, side='right')
    return H[idx]

def quantile_CI(cycle_len, result, batch_quantile_flag, tile, Q2_total, random:np.random.RandomState, return_L=False):
    assert len(result) == len(cycle_len)
    n_result = len(result)
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
                batch = result[index:min(index+batch_quantile,n_result)]
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
    if return_L:
        return Q2_total, section_std, batch_mean, batch_std, L1
    else:
        return Q2_total, section_std, batch_mean, batch_std

def quantile_var(L1, cycle_len, result, batch_quantile_flag, tile, random:np.random.RandomState, Q2_total):
    assert len(result) == len(cycle_len)
    n_result = len(result)
    np.random.seed(random.randint(100000))
    np.random.shuffle(cycle_len)
    batch_quantile = batch_quantile_flag
    batch_quantile = int(batch_quantile)
    for index in range(0, n_result, batch_quantile):
        batch = result[index:min(index+batch_quantile,n_result)]
        batch_tau = cycle_len[index:min(index+batch_quantile,n_result)]
        Q2 = sample_quantile(batch, batch_tau, tile)
        L1.append(Q2)
    batch_mean = np.mean(L1)
    batch_std = np.std(L1)/ np.sqrt(len(L1))
    section_std = np.sqrt(np.mean((np.array(L1)-Q2_total)**2)) / np.sqrt(len(L1))
    print(f'{Q2_total} sectioning CI ({Q2_total-1.96*section_std}, {Q2_total+1.96*section_std}) RE: {section_std/Q2_total}')
    print(f'{batch_mean} batching CI ({batch_mean-1.96*batch_std}, {batch_mean+1.96*batch_std}) RE: {batch_std/batch_mean}')

    return L1, batch_mean, batch_std, section_std

def continue_run(func_de, func_nom, cycle_len, result, L, n_cycle, batch_quantile_flag, tile, random):
    cycle_len1 = func_de(n_cycle)
    cycle_len.extend(cycle_len1)
    result1 = func_nom(n_cycle)
    result.extend(result1)
    Q2_total = sample_quantile(result, cycle_len, tile)
    L, batch_mean, batch_std, section_std = quantile_var(L, cycle_len1, result1, batch_quantile_flag, tile, random, Q2_total)
    print(f'cycles: {len(cycle_len)}')
    return cycle_len, result, Q2_total, section_std, L


def parameter():
    lam = np.array([10,10,10,400,200,100, 100, 450])*1e6
    size = np.array([100, 100, 200, 1400, 1400,1400, 1400,1400]) * 8
    size = size

    port_rate = 3e9
    arrival_rate = ((lam/size))
    theta = size / port_rate

    mu = 1/theta
    return arrival_rate/1e6, mu/1e6