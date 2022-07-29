from collections import defaultdict
import time

import numpy as np
import ray

from ns.environment import Environment
from ns.packet.dist_generators import DistPacketGenerators
from ns.packet.sink import PacketSink
from ns.scheduler.sp import SPServer
from ns.scheduler.wfq import WFQServer

def combine_result(results:dict, result_batch:list[dict]):
    for result in result_batch:
        for k,v in result.items():
            results[k].append(v)
    return results

class WeightedFairQueue:
    def __init__(self, 
                 num_type:int,
                 arrival_dists,
                 size_dists,
                 target_flow,
                 gamma,
                 port_rate=8,
                 seed:int = None,
                 record = True):
        self.num_type = num_type
        self.arrival_dists = arrival_dists
        self.size_dists = size_dists
        self.weights = {0:2, 1:1}
        self.port_rate = port_rate # 1 byte per second
        self.gamma = gamma
        self.target_flow = target_flow
        self.random = np.random.RandomState(seed)
        self.record = defaultdict(lambda: False, record)
        
    def simulate_one_cycle(self):
        # enviromnent with importance sampling
        env = Environment(self.num_type, self.gamma, self.target_flow)
        env.phase = 1
        env.reason = -1
        # element 
        wfq_server = WFQServer(env, self.port_rate, self.weights)
        ps = PacketSink(env, rec_flow_ids=True)
        pgs = [DistPacketGenerators(env, self.arrival_dists[flow], self.size_dists[flow], 2, flow_id=flow) for flow in range(self.num_type)]
        env.port_pgs = pgs 
        env.port_sp = wfq_server

        # connect
        for pg in pgs:
            pg.out = wfq_server
        wfq_server.out = ps
        
        # run
        env.cycle_run()

        # # statistics
        # if self.denominator:
        #     result = len(pgs[self.target_flow].packets)*np.exp(env.cycle_logW) 
        # else:
        #     result = sum([(packet.response_time>self.gamma)*np.exp(packet.response_llr) for packet in pgs[self.target_flow].packets])
        # statistics
        result = {}
        if self.record['num_packet']: # record the number of generated packets
            result['num_packet'] = sum([len(pgs[flow_idx].packets) for flow_idx in range(self.num_type)])
        if self.record['reason']:
            assert env.reason is not None
            result['reason'] = env.reason
        if self.record['cycle_len']:
            result['cycle_len'] = len(pgs[self.target_flow].packets)*np.exp(env.cycle_logW) 
        if self.record['response']:
            result['response'] = [packet.response_time for packet in pgs[self.target_flow].packets]
        if self.record['response_sum']:
            result['response_sum'] = sum([packet.response_time*np.exp(packet.response_llr) for packet in pgs[self.target_flow].packets])
        if self.record['prob_sum']:
            result['prob_sum'] = sum([(packet.response_time>self.gamma)*np.exp(packet.response_llr) for packet in pgs[self.target_flow].packets])
        if self.record['cross_entropy']:
            result['cross_entropy'] = env.CE_para
        if self.record['max_response']:
            result['max_response'] = max([packet.response_time for packet in pgs[self.target_flow].packets]+[0])    
        if self.record['save_response']:
            result['response'] = [(packet.response_time, packet.response_llr) for packet in pgs[self.target_flow].packets]

        return result    

    def simulate_cycles(self, num_cycle, start, mute=False):
        results = []
        for i in range(num_cycle):
            if not mute and i % 5000 == 1:
                print(f'{time.time()-start:.2f}s, {i}')
            results.append(self.simulate_one_cycle())
        return results
    
@ray.remote  
class WeightedFairQueue_ray(WeightedFairQueue):
    def __init__(self, num_type: int, arrival_dists, size_dists, target_flow, gamma, port_rate=8, seed: int = None, record=True):
        super().__init__(num_type, arrival_dists, size_dists, target_flow, gamma, port_rate, seed, record)
    
def static_wfq_sim(num_type, lam, mu, target_flow, gamma, port_rate, Distri, cpu_nums, n_cycles, start, mute=False, record = False):
    if cpu_nums is None or cpu_nums == 1:
        sim = WeightedFairQueue(num_type, 
                            [Distri(lam[i], np.random.randint(1000000)) for i in range(num_type)], 
                            [Distri(mu[i], np.random.randint(1000000)) for i in range(num_type)], 
                            target_flow=target_flow, gamma=gamma, 
                            port_rate=port_rate, record=record)
        result = sim.simulate_cycles(n_cycles, start, mute)
        results_combine = defaultdict(list)
        combine_result(results_combine, result)
    else:
        if not ray.is_initialized():
            ray.init(num_cpus=cpu_nums)
        sims = [WeightedFairQueue_ray.remote(num_type, 
                                        [Distri(lam[i], np.random.randint(1000000)) for i in range(num_type)], 
                                        [Distri(mu[i], np.random.randint(1000000)) for i in range(num_type)], 
                                        target_flow=target_flow, gamma=gamma, 
                                        port_rate=port_rate, record=record) for _ in range(cpu_nums)]
        result_id = [sim.simulate_cycles.remote(n_cycles // cpu_nums, start, mute) for sim in sims]
        results = ray.get(result_id)
        results_combine = defaultdict(list)
        for result in results:
            combine_result(results_combine, result)
        ray.shutdown()
    return results_combine