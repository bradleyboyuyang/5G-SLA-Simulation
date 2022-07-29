import time
import numpy as np
from distribution import ExponDistri
from simulation import static_priority_sim

def naive_cycle_length(lam, mu, port_rate, n_cycles, cpu_nums, num_type):
    start_local = time.time()
    lam = np.array(lam).reshape(-1, 1)
    mu = np.array(mu).reshape(-1, 1)
    thresholds = [[0] for _ in range(num_type)]

    print(f'(+{time.time()-start_local:.2f}s)Run simulation...')
    results = static_priority_sim(num_type, lam, mu, thresholds, 0, port_rate, ExponDistri, cpu_nums, n_cycles, start_local, p={0:1}, policy=None, mute=True, record={'cycle_len': True})
    cycle_len = results['cycle_len']
    print(f'(+{time.time()-start_local:.2f}s)End simulation.')
    
    L = []
    for flow_id in range(num_type):
        cycle_num = [result[flow_id] for result in cycle_len]
        mean = np.mean(cycle_num)
        halfCI = 1.96 * np.std(cycle_num) / np.sqrt(len(cycle_num))
        # L.append([mean, halfCI, halfCI/1.96/mean])
        L.append([mean, np.var(cycle_num)])
    return L

if __name__ == '__main__':
    np.random.seed(42)
    ## lam mu   original
    # num_type = 2
    # lam = [0.6, 0.2]
    # mu = [2, 1]

    # # # parameters
    # num_type = 4
    # lam = np.array([0.6, 0.2, 0.2, 0.2], dtype=float)
    # mu = np.array([2, 1, 2, 2], dtype=float)

    num_type = 8
    lam = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
    mu = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float)


    # p = lam / np.sum(lam)
    # rho = lam / mu
    # rho_total = np.sum(rho)
    # ES, ES2 = np.sum(p/mu), 2*np.sum(p/mu**2)
    # Se = ES2 / ES / 2
    # theory = [rho_total*Se/(1-rho[:flow_id+1].sum())/(1-rho[:flow_id].sum()) + 1/mu[flow_id] for flow_id in range(num_type)]
    # print(theory)

    port_rate = 1 * 8 # 1 byte per second
    print(naive_cycle_length(lam, mu, port_rate, 100000, 8, num_type))
