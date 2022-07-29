import time
from collections import defaultdict

import numpy as np
import ray

from environment import Environment
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.sink import PacketSink
from ns.scheduler.sp import SPServer
from utils import combine_result


class StaticPriority:
    def __init__(self, 
                 num_type:int,
                 arrival_dists,
                 size_dists,
                 threshold,
                 test_stop,
                 port_rate=8,
                 debug = False, 
                 seed:int = None,
                 policy:int = 0,
                 p:dict=None,
                 record = {'num_packet': True, 'cycle_len':True, 'response_sum':True, 'prob_sum':True, 'cross_entropy':True, 'max_response':True}):
        """One Stattion Static Priority Queue (Importance Sampling)

        Args:
            num_type (int): number of types of packets 
            arrival_dists (list of list of distribution): arrival distributions. 1d type 2d phase
            size_dists (list of list of distribution): service distributions. 1d type 2d phase
            threshold (list of list of float): gamma for each type of packets. 1d type 2d increasing
            test_stop (int): stop by which type
            port_rate (float, optional): port rate. Defaults to 8.
            debug (bool, optional): Debug or not. Defaults to False.
            policy (int, optional): policy for switching the measure. Defaults to 0.
            p (dict, optional): The prob of mixed method. Default is no.
        """        
        self.num_type = num_type
        self.arrival_dists = arrival_dists
        self.size_dists = size_dists
        self.priority = dict(zip(range(num_type), -np.array(range(num_type)))) # 0 > 1 > 2 > ...
        self.port_rate = port_rate # 1 byte per second
        self.debug = debug
        self.threshold = threshold
        self.record = defaultdict(lambda: False, record)
        self.test_stop = test_stop
        self.p = p
        self.policy = policy
        self.random = np.random.RandomState(seed)
        
    def simulate_one_cycle(self, axes=None):
        # enviromnent with importance sampling
        env = Environment(self.num_type, self.threshold, self.test_stop, self.random, policy=self.policy, p=self.p)
        env.add_likelihood(self.arrival_dists, self.size_dists)
        # element 
        sp_server = SPServer(env, self.port_rate, self.priority)
        env.add_port(sp_server)
        ps = PacketSink(env, rec_flow_ids=True)
        pgs = [DistPacketGenerator(env, f"flow_{flow}", self.arrival_dists[flow], self.size_dists[flow],
                                    flow_id=flow, rec_flow=True, debug=self.debug) for flow in range(self.num_type)]
        # connect
        for pg in pgs:
            pg.out = sp_server
        sp_server.out = ps
        # run
        env.cycle_run()

        # statistics
        result = {}
        if self.record['num_packet']: # record the number of generated packets
            result['num_packet'] = sum([len(env.packets[i]) for i in range(self.num_type)])
        if self.record['cycle_len']:
            result['cycle_len'] = [len(env.packets[flow])*np.exp(env.likelihood.logW) 
                                   for flow in range(self.num_type)]
        if self.record['response_sum']:
            result['response_sum'] = [sum([packet.packet_info.response*np.exp(packet.packet_info.llr_response)  
                                           for packet in env.packets[flow]]) 
                                      for flow in range(self.num_type)]
        if self.record['prob_sum']:
            result['prob_sum'] = [[sum([(packet.packet_info.response>gamma)*np.exp(packet.packet_info.llr_response)
                                        for packet in env.packets[flow]])
                                   for gamma in self.threshold[flow]]
                                  for flow in range(self.num_type)]
        if self.record['cross_entropy']:
            H = result['prob_sum'][env.stop][-1]
            CE = env.likelihood.CE * H
            result['cross_entropy'] = CE
        if self.record['max_response']:
            result['max_response'] = max([packet.packet_info.response for packet in env.packets[env.stop]]+[0])
            
        if self.record['reason']:
            result['reason'] = env.reason
            
        if self.record['save_response']:
            result['response'] = [(packet.packet_info.response, packet.packet_info.llr_response) 
                                  for packet in env.packets[env.stop]]
            result['CE'] = env.likelihood.CE

        return result    

    def simulate_cycles(self, num_cycle, start, mute=False):
        results = []
        for i in range(num_cycle):
            if not mute and i % 5000 == 1:
                print(f'{time.time()-start:.2f}s, {i}')
            results.append(self.simulate_one_cycle())
        return results

@ray.remote  
class StaticPriority_ray(StaticPriority):
    def __init__(self, num_type: int, arrival_dists, size_dists, threshold, test_stop, port_rate=8, debug=False, seed: int = None, policy: int = 0, p: dict = None, record={ 'num_packet': True,'cycle_len': True,'response_sum': True,'prob_sum': True,'cross_entropy': True,'max_response': True}):
        super().__init__(num_type, arrival_dists, size_dists, threshold, test_stop, port_rate, debug, seed, policy, p, record)

def static_priority_sim(num_type, lam, mu, threshold, test_stop, port_rate, Distri, cpu_nums, n_cycles, start, p=None, policy:int=0, mute=False, record = {'num_packet': True, 'cycle_len':True, 'response_sum':True, 'prob_sum':True, 'cross_entropy':True, 'max_response':True}):
    if cpu_nums is None or cpu_nums == 1:
        sim = StaticPriority(num_type, 
                            [Distri(lam[i], np.random.randint(1000000)) for i in range(num_type)], 
                            [Distri(mu[i], np.random.randint(1000000)) for i in range(num_type)], 
                            threshold, test_stop=test_stop,# seed=np.random.randint(1000000),
                            policy=policy,
                            port_rate=port_rate, p=p)
        result = sim.simulate_cycles(n_cycles, start, mute)
    else:
        if not ray.is_initialized():
            ray.init(num_cpus=cpu_nums)
        sims = [StaticPriority_ray.remote(num_type, 
                                        [Distri(lam[i], np.random.randint(1000000)) for i in range(num_type)], 
                                        [Distri(mu[i], np.random.randint(1000000)) for i in range(num_type)], 
                                        threshold, test_stop=test_stop, #seed=np.random.randint(1000000),
                                        policy = policy,
                                        port_rate=port_rate, p=p, record=record) for _ in range(cpu_nums)]
        result_id = [sim.simulate_cycles.remote(n_cycles // cpu_nums, start, mute) for sim in sims]
        results = ray.get(result_id)
        results_combine = defaultdict(list)
        for result in results:
            combine_result(results_combine, result)
        ray.shutdown()
    return results_combine
