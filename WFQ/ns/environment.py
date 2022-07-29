import numpy as np
import simpy 

class Environment(simpy.Environment):
    def __init__(self, num_type:int, gamma:float, target_flow:int):
        """My environment for importance sampling with regenerative method

        Args:
            num_type (int): the number of types / flows
            gamma (float): switching gamma for the target flow
            target_flow (int): stop by which type
        """        
        super().__init__()
        self.num_type = num_type
        self.target_flow = target_flow
        self.gamma = gamma

    def cycle_run(self):
        try:
            self.run(until=None)
        except ZeroDivisionError:
            pass

    def switch_CoE(self):
        assert self.phase != 0
        self.phase = 0
        for pg in self.port_pgs:
            age = pg.change_phase()
            if pg.flow_id == self.target_flow:
                assert np.isclose(age,0)
            
    def change_size(self, packet):
        # If the current phase phi is 0 but service phase is 1, re-sample service time: in order to reduce variance
        pre_size = packet.size
        assert self.phase == 0
        service = self.port_pgs[packet.flow_id].size_dist.sample(phase=0)
        packet.size = service
        packet.service_phase = self.phase
        packet.service_llr = 0
        return - pre_size + service   ##return the updated size

    def total_logW(self, service_llr):
        return sum([pg.arrival_logW() for pg in self.port_pgs]) + sum([service_llr[flow] for flow in range(self.num_type)])
        
    def ending(self, service_llr):
        self.cycle_logW = self.total_logW(service_llr)
        self.CE_para = self.CE()
        raise ZeroDivisionError # We abuse the error.
    
    def CE(self):
        arrival_num = np.zeros(self.num_type)
        arrival_sum = np.zeros(self.num_type)
        service_num = np.zeros(self.num_type)
        service_sum = np.zeros(self.num_type)
        for flow in range(self.num_type):
            arrival_num[flow], arrival_sum[flow] = self.port_pgs[flow].CE()
            service_num[flow] = self.port_sp.service_num[flow]
            service_sum[flow] = self.port_sp.service_sum[flow]
        result = np.array([arrival_num, arrival_sum, service_num, service_sum]) # 4 * K
        assert result.shape[0] == 4
        return result



if __name__ == '__main__':
    env = Environment(2, [[2],[4]], 1)
    env.cycle_run()
    print(env.now)
