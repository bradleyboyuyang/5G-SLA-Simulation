from collections import defaultdict as dd

import numpy as np

from distribution import Distribution


class PacketInfo:
    def __init__(self, packet_id:int, last_arrival:float, interarrival:float, arrival_phase:float, arrival_llr:float) -> None:
        self.packet_id = packet_id
        self.last_arrival = last_arrival
        self.interarrival = interarrival
        self.arrival_time = last_arrival + interarrival
        self.arrival_phase = arrival_phase
        self.arrival_llr = arrival_llr
        self.service = None
        self.start_service = None
        self.depart = None
        
    

class Likelihood:
    def __init__(self, env, num_type:int, arrival_dists:list[Distribution], size_dists:list[Distribution]):
        self.env = env
        self.num_type = num_type
        self.arrival_dists = arrival_dists
        self.size_dists = size_dists

        self.packets_info = dd(list)
        self.first_nonempty_time = None

    def put_interarrival(self, flow_id:int, packet_sent:int, last_arrival:float, interarrival:float, arrival_phase:int):
        if self.first_nonempty_time is None:
            assert np.isclose(last_arrival, 0)
            self.first_nonempty_time = interarrival
        assert len(self.packets_info[flow_id]) == packet_sent
        llr = self.arrival_dists[flow_id].logW(interarrival, phase=arrival_phase)
        packet_info = PacketInfo(packet_sent, last_arrival, interarrival, arrival_phase, llr)
        self.packets_info[flow_id].append(packet_info)
        return packet_info

    def put_service(self, flow_id:int, service_phase:int, service:float, packet_info:PacketInfo):
        packet_info.service_phase = service_phase
        packet_info.service = service
        packet_info.service_llr = self.size_dists[flow_id].logW(service, phase=service_phase)
        return packet_info

    def start_service(self, packet):
        packet_info = packet.packet_info
        self.env.current_packet = packet
        packet_info.start_service = self.env.now
        packet_info.wait = packet_info.start_service - packet_info.arrival_time
        packet_info.llr_wait = self.logW(self.env.n_served, packet.flow_id)
        return packet_info

    def depart_service(self, packet):
        packet_info = packet.packet_info
        packet_info.depart = self.env.now 
        packet_info.response = packet_info.wait + packet_info.service
        packet_info.llr_response = packet_info.llr_wait + packet_info.service_llr

    def change_size(self, packet):
        packet_info = packet.packet_info
        if packet_info.service_phase != self.env.phase:
            if self.env.critical_packet.start_service is not None and self.env.now > self.env.critical_packet.start_service:
                pre_size = packet.size
                service = self.size_dists[packet.flow_id].sample(phase=self.env.phase)
                packet.size = service
                self.put_service(packet.flow_id, self.env.phase, service, packet_info)
                return - pre_size + service
        return None

    # sum up the log likelihood ratios of the previously served packets (A and S). It returns the log likehood ratio that the current packet is served at time s. The calculation can be divided into three cases (higher/same/lower priority).
    def logW(self, n_served, flow_id):
        logW = 0
        for flow in range(self.num_type):
            packet_infos = self.packets_info[flow]
            # arrival
            if flow < flow_id: ## higher priority, consider tail
                arrival_llrs = [packet_info.arrival_llr for packet_info in packet_infos[:n_served[flow]]]
                logW += sum(arrival_llrs) + self.logW_tail(flow, n_served[flow], self.env.now)
            elif flow == flow_id: 
                arrival_llrs = [packet_info.arrival_llr for packet_info in packet_infos[:n_served[flow]+1]]
                logW += sum(arrival_llrs) # privoius + current
            else: ## lower priority, consider initial case
                if n_served[flow] == 0:
                    logW += self.logW_tail(flow, 0, self.first_nonempty_time)
                else:
                    arrival_llrs = [packet_info.arrival_llr for packet_info in packet_infos[:n_served[flow]]]
                    logW += sum(arrival_llrs)
            # service
            service_llrs = [packet_info.service_llr for packet_info in packet_infos[:n_served[flow]]]
            logW += sum(service_llrs)
        return logW

    # sum up the log likelihood ratios of all the packets (A and S) in the cycle. The result is the log likelihood ratio of this cycle. Used for adjusting the length of cycles
    def cycle_logW(self, n_served):
        logW = 0
        for flow in range(self.num_type):
            packet_infos = self.packets_info[flow]
            arrival_llrs = [packet_info.arrival_llr for packet_info in packet_infos[:n_served[flow]]]
            logW += sum(arrival_llrs) + self.logW_tail(flow, n_served[flow], self.env.now)
            service_llrs = [packet_info.service_llr for packet_info in packet_infos[:n_served[flow]]]
            logW += sum(service_llrs)
        return logW

    # return the log likelihood ratio of the next packet arrives later than current time s, used for calculating the higher priority case
    def logW_tail(self, flow_id, idx, now):
        packet_info = self.packets_info[flow_id][idx]
        # return packet_info.arrival_llr
        phase = packet_info.arrival_phase
        last_arrival = packet_info.last_arrival
        return self.arrival_dists[flow_id].logW_tail(now - last_arrival, phase=phase)

    # return a 4*K list recording the arrival_num, arrival_sum, service_num and service_sum of each type during the phi=1 period
    def cycle_CE(self, n_served):
        arrival_num = np.zeros(self.num_type)
        arrival_sum = np.zeros(self.num_type)
        service_num = np.zeros(self.num_type)
        service_sum = np.zeros(self.num_type)
        for flow in range(self.num_type):
            assert len(self.packets_info[flow]) == n_served[flow] + 1
            for packet in self.packets_info[flow]:
                if packet.arrival_phase == 1:
                    arrival_num[flow] += 1
                    arrival_sum[flow] += packet.interarrival
            for packet in self.packets_info[flow][:-1]:
                if packet.service_phase == 1:
                    service_num[flow] += 1
                    service_sum[flow] += packet.service
        result = np.array([arrival_num, arrival_sum, service_num, service_sum]) # 4 * K
        return result
            
class Likelihood_mix(Likelihood):
    def __init__(self, env, num_type: int, arrival_dists: list[Distribution], size_dists: list[Distribution]):
        super().__init__(env, num_type, arrival_dists, size_dists)
        self.p = env.p

    def put_interarrival(self, flow_id: int, packet_sent: int, last_arrival: float, interarrival: float, arrival_phase: int):
        packet_info = super().put_interarrival(flow_id, packet_sent, last_arrival, interarrival, arrival_phase)
        if self.p is not None:
            if arrival_phase == 0:
                packet_info.llr_arrival_mix = {k:0 for k in self.p}
            else:
                phases = list(self.p.keys())
                packet_info.llr_arrival_mix = dict(zip(phases, self.arrival_dists[flow_id].logW(interarrival, phase=phases))) 
        return packet_info

    def put_service(self, flow_id: int, service_phase: int, service: float, packet_info: PacketInfo):
        packet_info = super().put_service(flow_id, service_phase, service, packet_info)
        if self.p is not None:
            if service_phase == 0:
                packet_info.llr_service_mix = {k:0 for k in self.p}
            else:
                phases = list(self.p.keys())
                packet_info.llr_service_mix = dict(zip(phases, self.size_dists[flow_id].logW(service, phase=phases))) 
        return packet_info

    def start_service(self, packet):
        packet_info = super().start_service(packet)
        if self.p is not None:
            packet_info.llr_waits = self.logW_mix(self.env.n_served, packet.flow_id)
        return packet_info

    def logW_mix(self, n_served, flow_id):
        logWs = {}
        for phase_CoE in self.p.keys():
            logW = 0
            for flow in range(self.num_type):
                packet_infos = self.packets_info[flow]
                # arrival
                if flow < flow_id: ## higher priority, consider tail
                    arrival_llrs = [packet_info.llr_arrival_mix[phase_CoE] for packet_info in packet_infos[:n_served[flow]]]
                    logW += sum(arrival_llrs) + self.logW_tail_mix(flow, n_served[flow], self.env.now, phase_CoE)
                elif flow == flow_id: 
                    arrival_llrs = [packet_info.llr_arrival_mix[phase_CoE] for packet_info in packet_infos[:n_served[flow]+1]]
                    logW += sum(arrival_llrs) # privoius + current
                else: ## lower priority, consider initial case
                    if n_served[flow] == 0:
                        logW += self.logW_tail_mix(flow, 0, self.first_nonempty_time, phase_CoE)
                    else:
                        arrival_llrs = [packet_info.llr_arrival_mix[phase_CoE] for packet_info in packet_infos[:n_served[flow]]]
                        logW += sum(arrival_llrs)
                # service
                service_llrs = [packet_info.llr_service_mix[phase_CoE] for packet_info in packet_infos[:n_served[flow]]]
                logW += sum(service_llrs)
            logWs[phase_CoE] = logW
        return logWs

    def cycle_logW_mix(self, n_served):
        logWs = {}
        for phase_CoE in self.p.keys():
            logW = 0
            for flow in range(self.num_type):
                packet_infos = self.packets_info[flow]
                arrival_llrs = [packet_info.llr_arrival_mix[phase_CoE] for packet_info in packet_infos[:n_served[flow]]]
                logW += sum(arrival_llrs) + self.logW_tail_mix(flow, n_served[flow], self.env.now, phase_CoE)
                service_llrs = [packet_info.llr_service_mix[phase_CoE] for packet_info in packet_infos[:n_served[flow]]]
                logW += sum(service_llrs)
            logWs[phase_CoE] = logW
        return logWs

    def logW_tail_mix(self, flow_id, idx, now, phase_CoE):
        packet_info = self.packets_info[flow_id][idx]
        if packet_info.arrival_phase == 0:
            return 0
        else:
            # return packet_info.arrival_llr
            last_arrival = packet_info.last_arrival
            return self.arrival_dists[flow_id].logW_tail(now - last_arrival, phase=phase_CoE)
