"""
Implements a packet generator that simulates the sending of packets with switching distributions.
Specifically, first we send the phase 1 packets and then send the phase 0 packets after calling the change_phase method.
"""
from functools import partial

import numpy as np
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.packet import Packet
from ns.packet.distribution import Distribution

class DistPacketGenerators:
    def __init__(self,
                 env,
                 arrival_dist:Distribution,
                 size_dist:Distribution,
                 num_phase:int=2,
                 initial_delay=0,
                 finish=float("inf"),
                 flow_id=0,
                 debug=False):
        self.env = env
        self.debug = debug
        self.arrival_dist = arrival_dist
        self.size_dist = size_dist
        self.flow_id = flow_id
        self.phase = self.env.phase
        
        # distribution
        arrival_dists = [partial(self.arrival_dist.sample, phase) for phase in range(num_phase)]
        size_dists = [partial(self.size_dist.sample, phase) for phase in range(num_phase)]
        
        # a group of dist_generators
        self.pgs = [DistPacketGenerator(env, phase, arrival_dists[phase], size_dists[phase], initial_delay=initial_delay, finish=finish, flow_id=flow_id, rec_flow=False, debug=False) 
                    for phase in range(num_phase)]  # packet.src is arrival phase
        self.pgs[self.env.phase].debug = self.debug
        for pg in self.pgs:
            pg.out = self 
            
        self.packets = []
            
        # structure
        self.out = None
        # likelihood ratio
        self.last_arrival_time = 0
        self.arrival_llr = 0
        self.phase_age = None
        self.stopping_time = None
        # cross-entropy
        self.sum_A = 0
        self.num_A = 0
        
                
    def put(self, packet:Packet):
        if packet.src == self.env.phase:
            packet.service_phase = packet.src
            self.packets.append(packet)
            if self.env.phase == 1: # IS phase
                interarrival = self.env.now - self.last_arrival_time
                self.num_A += 1
                self.sum_A += interarrival
                self.arrival_llr += self.arrival_dist.logW(interarrival, phase=1)
                packet.service_llr = self.size_dist.logW(packet.size, phase=1)
            self.last_arrival_time = self.env.now
            self.out.put(packet)
            
    def change_phase(self):
        assert self.phase == 1
        if self.debug:
            print(f'change the phase from 1 to 0 at {self.env.now} and the last arrival is {self.env.now - self.last_arrival_time} ago')
            self.pgs[1].debug = False
            self.pgs[0].debug = True
        self.phase = 0 
        self.phase_age = self.env.now - self.last_arrival_time
        self.phase_tail_llr = self.arrival_dist.logW_tail(self.phase_age, phase=1)
        self.stopping_time = self.env.now
        return self.phase_age
    
    def arrival_logW(self):
        if self.phase == 1:
            assert self.stopping_time is None 
            return self.arrival_llr + self.arrival_dist.logW_tail(self.env.now - self.last_arrival_time, phase=1)
        else:
            return self.arrival_llr + self.phase_tail_llr
    
    def CE(self):
        if self.phase_age is None:
            assert self.phase == 1 and self.stopping_time is None
            self.stopping_time = self.env.now
            self.phase_age = self.env.now - self.last_arrival_time
        assert np.isclose(self.sum_A + self.phase_age, self.stopping_time), (self.sum_A + self.phase_age, self.stopping_time)
        return self.num_A, self.stopping_time