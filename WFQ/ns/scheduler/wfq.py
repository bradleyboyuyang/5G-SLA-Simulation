"""
Implements a Weighted Fair Queueing (WFQ) server.
Reference:
Class-Based Weighted Fair Queueing,
https://www.cisco.com/en/US/docs/ios/12_0t/12_0t5/feature/guide/cbwfq.html#wp17641
"""
from collections import defaultdict as dd
from collections.abc import Callable

from ns.packet.packet import Packet
from ns.utils import taggedstore


class WFQServer:
    """
    Parameters
    ----------
    env: simpy.Environment
        The simulation environment.
    rate: float
        The bit rate of the port.
    weights: list or dict 
        This can be either a list or a dictionary. If it is a list, it uses the flow_id ---
        or class_id, if class-based fair queueing is activated using the `flow_classes' parameter
        below --- as its index to look for the flow (or class)'s corresponding weight. If it is a
        dictionary, it contains (flow_id or class_id -> weight) pairs for each possible flow_id
        or class_id.
    flow_classes: function
        This is a function that matches flow_id's to class_ids, used to implement class-based
        Weighted Fair Queueing. The default is an identity lambda function, which is equivalent
        to flow-based WFQ.
    zero_buffer: bool
        Does this server have a zero-length buffer? This is useful when multiple
        basic elements need to be put together to construct a more complex element
        with a unified buffer.
    zero_downstream_buffer: bool
        Does this server's downstream element has a zero-length buffer? If so, packets
        may queue up in this element's own buffer rather than be forwarded to the
        next-hop element.
    debug: bool
        If True, prints more verbose debug information.
    """
    def __init__(self,
                 env,
                 rate: float,
                 weights,
                 flow_classes: Callable = lambda x: x,
                 zero_buffer=False,
                 zero_downstream_buffer=False,
                 debug: bool = False) -> None:
        self.env = env
        self.rate = rate
        self.weights = weights

        self.flow_classes = flow_classes
        # 记录结束时间
        self.finish_times = {}
    
        self.flow_queue_count = {}

        if isinstance(weights, list):
            for queue_id in range(len(weights)):
                self.finish_times[queue_id] = 0.0
                self.flow_queue_count[queue_id] = 0

        elif isinstance(weights, dict):
            for (queue_id, __) in weights.items():
                self.finish_times[queue_id] = 0.0
                self.flow_queue_count[queue_id] = 0
        else:
            raise ValueError('Weights must be either a list or a dictionary.')

        self.active_set = set() # active flow
        self.vtime = 0.0 # virtual time
        self.out = None 
        self.packets_received = 0
        self.packets_dropped = 0
        self.debug = debug
        self.count = 0


        self.current_packet = None
        # 如果default dict中不存在key就返回0
        self.byte_sizes = dd(lambda: 0)

        self.upstream_updates = {}
        self.upstream_stores = {}
        self.zero_buffer = zero_buffer
        self.zero_downstream_buffer = zero_downstream_buffer
        if self.zero_downstream_buffer:
            self.downstream_store = taggedstore.TaggedStore(env)

        self.store = taggedstore.TaggedStore(env)
        self.action = env.process(self.run())
        self.last_update = 0.0

        self.service_llr = dd(int) # 字典不存在就返回0
        self.service_num = dd(int) # cycle num
        self.service_sum = dd(int) # numerator
        self.last_class = None


    def update_stats(self, packet):
        """
        The packet has been sent (or authorized to be sent if the downstream node has a zero-buffer
        configuration), we need to update the internal statistics related to this event.
        """
        flow_id = packet.flow_id

        self.flow_queue_count[self.flow_classes(flow_id)] -= 1

        # 传输完毕去除任务
        if self.flow_queue_count[self.flow_classes(flow_id)] == 0:
            self.active_set.remove(self.flow_classes(flow_id))

        self.last_update = self.env.now

        if len(self.active_set) == 0:
            self.vtime = 0.0
            for (queue_id, __) in self.finish_times.items():
                self.finish_times[queue_id] = 0.0

        if self.flow_classes(flow_id) in self.byte_sizes:
            self.byte_sizes[self.flow_classes(flow_id)] -= packet.size
        else:
            raise ValueError("Error: the packet is from an unrecorded flow.")

        if self.debug:
            print(f"Sent Packet {packet.packet_id} from flow {flow_id} "
                  f"belonging to class {self.flow_classes(flow_id)}")


    def update(self, packet):
        """
        The packet has just been retrieved from this element's own buffer by a downstream
        node that has no buffers. Propagate to the upstream if this node also has a zero-buffer
        configuration.
        """
        # With no local buffers, this element needs to pull the packet from upstream
        if self.zero_buffer:
            # For each packet, remove it from its own upstream's store
            self.upstream_stores[packet].get()
            del self.upstream_stores[packet]
            self.upstream_updates[packet](packet)
            del self.upstream_updates[packet]


    def packet_in_service(self) -> Packet:
        """
        Returns the packet that is currently being sent to the downstream element.
        Used by a ServerMonitor.
        """
        return self.current_packet

    def byte_size(self, queue_id) -> int:
        """
        Returns the size of the queue for a particular queue_id, in bytes.
        Used by a ServerMonitor.
        """
        if queue_id in self.byte_sizes:
            return self.byte_sizes[queue_id]

        return 0

    def size(self, queue_id) -> int:
        """
        Returns the size of the queue for a particular queue_id, in the
        number of packets. Used by a ServerMonitor.
        """
        return self.flow_queue_count[queue_id]

    def all_flows(self) -> list:
        """
        Returns a list containing all the queue IDs, which may be flow IDs or class IDs
        (in the case of class-based WFQ).
        """
        return self.byte_sizes.keys()


    def run(self):
        """The generator function used in simulations."""
        while True:
            if self.zero_downstream_buffer:
                packet = yield self.downstream_store.get()
                self.current_packet = packet
                yield self.env.timeout(packet.size * 8.0 / self.rate)

                self.update_stats(packet)
                self.out.put(packet,
                             upstream_update=self.update,
                             upstream_store=self.store)
                self.current_packet = None
            else:
                packet = yield self.store.get()

                self.current_packet = packet
                packet.start_service = self.env.now
                if packet.flow_id > self.env.target_flow and packet.src != self.env.phase:
                    change = self.env.change_size(packet)
                    self.byte_sizes[self.flow_classes(packet.flow_id)] += change

                # 处理change of measure的相关统计量
                if packet.service_phase == 1:
                    self.service_llr[packet.flow_id] += packet.service_llr
                    self.service_num[packet.flow_id] += 1
                    self.service_sum[packet.flow_id] += packet.size
                # response time
                packet.response_time = self.env.now + packet.size - packet.time
                # response likelihood
                packet.response_llr = self.env.total_logW(self.service_llr)
                # assert self.env.reason is not None
                # if packet.flow_id == self.env.target_flow and self.env.reason <= -1:
                #     if packet.response_time > self.env.gamma:
                #         if self.last_class is None:
                #             self.env.reason = 100
                #         else:
                #             self.env.reason = self.last_class

                yield self.env.timeout(packet.size * 8.0 / self.rate)

                self.update_stats(packet)
                self.update(packet)
                self.out.put(packet)
                self.current_packet = None

                total_size = sum(self.byte_sizes.values())
                if abs(total_size) < 1e-7: # stop the cycle
                    self.env.ending(self.service_llr)

                # total_size = sum(self.byte_sizes.values())
                # if abs(total_size) < 1e-7: # stop the cycle
                #     self.count += 1
                # if self.count == 100000:
                #     self.env.ending(self.service_llr)
                    

    def put(self, packet, upstream_update=None, upstream_store=None):
        """ Sends a packet to this element. """
        self.packets_received += 1
        flow_id = packet.flow_id

        self.byte_sizes[self.flow_classes(flow_id)] += packet.size
        now = self.env.now
        self.flow_queue_count[self.flow_classes(flow_id)] += 1
        self.active_set.add(self.flow_classes(flow_id))

        weight_sum = 0.0
        for i in self.active_set:
            weight_sum += self.weights[i]

        self.vtime += (now - self.last_update) / weight_sum
        # 最关键的公式
        self.finish_times[self.flow_classes(flow_id)] = max(
            self.finish_times[self.flow_classes(flow_id)], self.vtime
        ) + packet.size * 8.0 / self.weights[self.flow_classes(flow_id)]


        # if (self.env.phase != 0) and (flow_id == self.env.target_flow): # stopping time
        #     workload = sum([self.byte_sizes[self.flow_classes(flow_idx)] for flow_idx in range(self.env.target_flow+1)])
        #     # current packet
        #     if self.current_packet is not None:
        #         if self.current_packet.flow_id <= flow_id:
        #             workload -= (self.env.now - self.current_packet.start_service) # delete served bytes: bytes that the currently serving packet has been served
        #         else:
        #             workload += (self.current_packet.start_service + self.current_packet.size - self.env.now) # add remaining: the remaing bytes that the current serving packet would countinue to be served
        #     if workload > self.env.gamma:
        #         self.env.switch_CoE()

        if self.debug:
            print(
                f"Packet arrived at {self.env.now}, with flow_id {flow_id}, "
                f"belonging to class {self.flow_classes(flow_id)}, "
                f"packet_id {packet.packet_id}, "
                f"finish_time {self.finish_times[self.flow_classes(flow_id)]}")

        self.last_update = now

        if self.zero_buffer and upstream_update is not None and upstream_store is not None:
            self.upstream_stores[packet] = upstream_store
            self.upstream_updates[packet] = upstream_update

        if self.zero_downstream_buffer:
            self.downstream_store.put(
                (self.finish_times[self.flow_classes(flow_id)], packet))

        return self.store.put(
            (self.finish_times[self.flow_classes(flow_id)], packet))