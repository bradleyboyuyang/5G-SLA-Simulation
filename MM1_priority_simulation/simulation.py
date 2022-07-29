import time
import numpy as np
from environment import Environment
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.sink import PacketSink
from ns.scheduler.sp import SPServer
import ray 

 
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
                 p:dict=None):
        """One Station Static Priority Queue (Importance Sampling)

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
        # cycle_num: denominator estimate, i.e. the cycle length for each type of event in the current cycle
        # cycle_sum: numerator estimate for expected waiting time, i.e. the numerator for each type of event in one cycle
        # cycle_sum_prob: numerator estimate for the probability of exceeding the threshold (the same size as 'threshold')
        # reason: the type of event that first exceeds the threshold
        # num: number of packets in this cycle
        # max_wait: longest waiting time of all the packet in the cycle
        num = sum([len(env.packets[i]) for i in range(self.num_type)])
        cycle_num = np.zeros(self.num_type)
        cycle_sum = np.zeros(self.num_type)
        cycle_sum_prob = [np.zeros(len(self.threshold[i])) for i in range(self.num_type)]
        for flow in range(self.num_type):
            for packet in env.packets[flow]:
                packet_info = packet.packet_info
                cycle_num[flow] += 1 * np.exp(env.likelihood.logW)
                cycle_sum[flow] += packet_info.response * np.exp(packet_info.llr_response)
                for idx, gamma in enumerate(self.threshold[flow]):
                    if self.p is None:
                        cycle_sum_prob[flow][idx] += (packet_info.wait > gamma) * np.exp(packet_info.llr_wait)       
                    else: 
                        llr_waits = packet_info.llr_waits.values()
                        p = [self.p[phase] for phase in packet_info.llr_waits.keys()]
                        logWs = np.array(list(llr_waits))
                        Ws_inverse = np.exp(-logWs)
                        denom = np.inner(Ws_inverse, p)
                        cycle_sum_prob[flow][idx] += (packet_info.wait > gamma) * 1 / denom
        H = cycle_sum_prob[env.stop][-1]
        CE = env.likelihood.CE * H
        max_wait = max([packet.packet_info.wait for packet in env.packets[env.stop]]+[0]) 
        return cycle_num, cycle_sum, cycle_sum_prob, env.reason, num, CE.reshape(self.num_type*4), max_wait
        # return np.concatenate([cycle_num, cycle_sum] + cycle_sum_prob + env.reason)

    def simulate_cycles(self, num_cycle, start):
        results = []
        for i in range(num_cycle):
            if i % 10000 == 1:
                print(f'{time.time()-start:.2f}s, {i}')
            results.append(self.simulate_one_cycle())
        return results

@ray.remote  
class StaticPriority_ray(StaticPriority):
    def __init__(self, num_type: int, arrival_dists, size_dists, threshold, test_stop, port_rate=8, debug=False, seed:int=None, policy:int=0, p:dict=None):
        super().__init__(num_type, arrival_dists, size_dists, threshold, test_stop, port_rate, debug, seed, policy, p)

def static_priority_sim(num_type, lam, mu, threshold, test_stop, port_rate, Distri, cpu_nums, n_cycles, start, p=None, policy:int=0):
    if cpu_nums is None or cpu_nums == 1:
        sim = StaticPriority(num_type, 
                            [Distri(lam[i], np.random.randint(1000000)) for i in range(num_type)], 
                            [Distri(mu[i], np.random.randint(1000000)) for i in range(num_type)], 
                            threshold, test_stop=test_stop,# seed=np.random.randint(1000000),
                            policy=policy,
                            port_rate=port_rate, p=p)
        result = sim.simulate_cycles(n_cycles, start)
    else:
        sims = [StaticPriority_ray.remote(num_type, 
                                        [Distri(lam[i], np.random.randint(1000000)) for i in range(num_type)], 
                                        [Distri(mu[i], np.random.randint(1000000)) for i in range(num_type)], 
                                        threshold, test_stop=test_stop, #seed=np.random.randint(1000000),
                                        policy = policy,
                                        port_rate=port_rate, p=p) for _ in range(cpu_nums)]
        result_id = [sim.simulate_cycles.remote(n_cycles // cpu_nums, start) for sim in sims]
        results = ray.get(result_id)
        result = sum(results, [])
    resultT = list(zip(*result))
    return resultT