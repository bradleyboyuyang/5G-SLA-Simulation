import queue
import simpy 
import copy 
import numpy as np

from collections import defaultdict as dd

from likelihood_ratio import Likelihood_mix
from distribution import Distribution


class Environment(simpy.Environment):
    def __init__(self, num_type:int, gamma:list[list[float]], stop:int, random:np.random.RandomState, p:dict=None, policy=0):
        """My environment for importance sampling with regenerative method

        Args:
            num_type (int): the number of types / flows
            gamma (list[list[float]]): gamma for each type of packets
            stop (int): stop by which type
            p (list[float], optional): the prob for mixed measure method. Defaults to None.         
            policy (int, optional): policy for switching the measure. Defaults to 0.
        """        
        super().__init__()
        self.num_type = num_type
        self.stop = stop
        self.gamma = copy.deepcopy(gamma)
        self.policy = policy
        # threshold for stopping
        self.threshold = [max(self.gamma[i]) for i in range(num_type)] 
        # Reason for each threshold, the shape is the same as threshold. Defaults to -1.
        self.reason = [np.full(len(gamma[i]), -1) for i in range(num_type)]
        # the index of gamma need to be computed
        self.init_gamma = np.zeros(num_type, dtype=int)
        # Change of Measure by using which phase
        if p is None:
            self.p = p
            self.phase = 1
        else:
            self.p = {x:y for x,y in p.items() if y!=0} # delete 0 prob
            self.phase = random.choice(list(p.keys()), p=list(p.values()))
        self.init_phase = self.phase
        # record how to change the phase
        self.change_type = None
        # Collect the number of packets served in the station
        self.n_served = np.zeros(num_type, dtype=int)  ##record the number of served packets for each type
        self.last_flow = None  ## Mark the packet right before the system exceeds the threshold for the first time.
        self.current_packet = None
        # Mark the critical packet when turning to the original change of measure.
        self.critical_packet = None
        # collect packets
        self.packets = dd(list)

    def cycle_run(self):
        try:
            self.run(until=None)
        except ZeroDivisionError:
            pass

    def add_likelihood(self, arrival_dists:list[Distribution], size_dists:list[Distribution]):
        self.likelihood = Likelihood_mix(self, self.num_type, arrival_dists, size_dists)

    def add_port(self, station):
        self.port = station

    def current_workload(self, flow_id:int, packet_id:int):
        # return the current workload (including this packet) of packet `packet_id` of flow `flow_id`, which is a lower bound of response time.
        if self.current_packet is None:
            return 0
        else:
            packets = self.packets
            workload = 0
            # higher priority
            for flow in range(flow_id):
                for packet in packets[flow]:
                    if packet.packet_info.depart is None:
                        workload += packet.size
            # same priority
            for packet in packets[flow_id]:
                if packet.packet_info.depart is None and packet.packet_id <= packet_id: # response time
                    workload += packet.size
            # current packet
            if self.current_packet.flow_id <= flow_id:
                workload -= (self.now - self.current_packet.packet_info.start_service) # delete served bytes: bytes that the currently serving packet has been served
            else:
                workload += (self.current_packet.packet_info.start_service + self.current_packet.size - self.now) # add remaining: the remaing bytes that the current serving packet would countinue to be served
            return workload

    def check_gamma(self, packet, flow_id):
        init_gamma = self.init_gamma[flow_id]
        for idx in range(init_gamma, len(self.gamma[flow_id])):
            if packet.wait > self.gamma[flow_id][idx]:
                assert self.reason[flow_id][idx] == -1
                self.reason[flow_id][idx] = self.last_flow ## Mark the packet right before the system exceeds the threshold for the first time.
                # self.env.reason[packet.flow_id][idx] = packet.packet_id
                self.gamma[flow_id][idx] = np.inf ## since we only record the first time
                assert self.init_gamma[flow_id] == idx
                self.init_gamma[flow_id] += 1
            else:
                break

    def turning_point(self, flow_id, packet_id, packet_info):
        # If the packet does not have the highest priority, we need to check the workload when any a packet with higher priority arrives.
        if self.phase != 0:
            if flow_id == self.stop: ##current arrival is the target: k=k'
                if self.current_workload(flow_id, packet_id) > self.threshold[self.stop]: ##the lower bound already exceeds the threshold
                    self.phase = 0  ##return to the original measure
                    self.change_type == 'arrival'   ##Swithcing 1
                    assert self.critical_packet is None
                    self.critical_packet = packet_info   ##Mark the critical packet when turning to the original measure.
            elif flow_id < self.stop:  ##current arrival has a higher priority: k<k'
                policy = self.policy  ##different policy of switching measure
                if policy == 0:    ##Swithcing 1
                    pass 
                elif policy == 1:   ##Swithcing 2
                    if self.n_served[self.stop] < len(self.packets[self.stop]):
                        packet = self.packets[self.stop][self.n_served[self.stop]]
                        if self.current_workload(packet.flow_id, packet.packet_id) > self.threshold[self.stop]:  ##change the phase to 0 if the lower bound of the following packet of flow k' exceeds the threshold
                            self.phase = 0
                            self.change_type == 'higher arrival'
                            assert self.critical_packet is None
                            self.critical_packet = packet.packet_info
                elif policy == 2:   ##Swithcing 3
                    for packet in self.packets[self.stop][self.n_served[self.stop]:]:
                        if self.current_workload(packet.flow_id, packet.packet_id) > self.threshold[self.stop]:  ##change the phase to 0 if there exists a packet of flow k' whose lower bound exceeds the threshold
                            self.phase = 0
                            self.change_type == 'higher arrival'
                            assert self.critical_packet is None
                            self.critical_packet = packet.packet_info
                            break
        
    def ending(self):
        self.likelihood.logW = self.likelihood.cycle_logW(self.n_served)
        self.likelihood.CE = self.likelihood.cycle_CE(self.n_served)
        if self.p is not None:
            self.likelihood.logWs = self.likelihood.cycle_logW_mix(self.n_served)
        raise ZeroDivisionError # We abuse the error.



if __name__ == '__main__':
    env = Environment(2, [[2],[4]], 1)
    env.cycle_run()
    print(env.now)
