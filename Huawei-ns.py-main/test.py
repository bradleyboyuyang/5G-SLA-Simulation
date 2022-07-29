"""
An example of using the Weighted Fair Queueing (WFQ) scheduler.
"""
from functools import partial
from random import expovariate

import matplotlib.pyplot as plt
import simpy
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.sink import PacketSink
from ns.scheduler.monitor import ServerMonitor
from ns.scheduler.wfq import WFQServer
from ns.utils.splitter import Splitter
from ns.packet.distribution import ExponDistri
import numpy as np


num_type = 2
lam = [0.2, 0.4]
mu = [2, 1]
weights = {0:2, 1:1}

env = simpy.Environment()

# arrival_dists = [ExponDistri(lam[i], np.random.randint(1000000)) for i in range(num_type)]
# size_dists = [ExponDistri(mu[i], np.random.randint(1000000)) for i in range(num_type)]

def arrival1():
    return np.random.exponential(scale=1/0.2, size=1)[0]

def arrival2():
    return np.random.exponential(scale=1/0.4, size=1)[0]

def arrival3():
    return np.random.exponential(scale=1/2, size=1)[0]

def arrival4():
    return np.random.exponential(scale=1/1, size=1)[0]    
pg1 = DistPacketGenerator(env,
                          "flow_0",
                          arrival1,
                          arrival3,
                          initial_delay=0.0,
                          finish=50,
                          flow_id=0)
pg2 = DistPacketGenerator(env,
                          "flow_1",
                          arrival2,
                          arrival4,
                          initial_delay=0,
                          finish=50,
                          flow_id=1)

# [pg1, pg2] = [DistPacketGenerators(env, arrival_dists[flow], size_dists[flow], 2, flow_id=flow) for flow in range(num_type)]

ps = PacketSink(env)
sink_1 = PacketSink(env)
sink_2 = PacketSink(env)

wfq_server = WFQServer(env, 8, [2,1])

monitor = ServerMonitor(env,
                        wfq_server,
                        partial(expovariate, 0.1),
                        pkt_in_service_included=True)
splitter_1 = Splitter()
splitter_2 = Splitter()

pg1.out = splitter_1
pg2.out = splitter_2

splitter_1.out1 = wfq_server
splitter_1.out2 = sink_1
splitter_2.out1 = wfq_server
splitter_2.out2 = sink_2

wfq_server.out = ps

env.run(until=1000)


# result = {}
# result['num_packet'] = sum([len(pg1.packets), len(pg2.packets)])
# print(result)
# result['response_sum'] = sum([packet.response_time*np.exp(packet.response_llr) for packet in pgs[self.target_flow].packets])



print("At the WFQ server, the queue lengths in # packets for flow 0 are:")
print(monitor.sizes[0])
print("At the WFQ server, the queue lengths in # packets for flow 1 are:")
print(monitor.sizes[1])
print("At the WFQ server, the queue lengths in bytes for flow 0 are:")
print(monitor.byte_sizes[0])
print("At the WFQ server, the queue lengths in bytes for flow 1 are:")
print(monitor.byte_sizes[1])

print("At the packet sink, packet arrival times for flow 0 are:")
print(ps.arrivals[0])

print("At the packet sink, packet arrival times for flow 1 are:")
print(ps.arrivals[1])

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
ax1.vlines(sink_1.arrivals[0],
           0.0,
           1.0,
           colors="g",
           linewidth=2.0,
           label='Flow 0')
ax1.vlines(sink_2.arrivals[1],
           0.0,
           0.7,
           colors="r",
           linewidth=2.0,
           label='Flow 1')
ax1.set_title("Arrival times at the WFQ server")
ax1.set_ylim([0, 1.5])
ax1.set_xlim([0, max(sink_1.arrivals[0]) + 10])
ax1.grid(True)
ax1.legend()

ax2.vlines(ps.arrivals[0], 0.0, 1.0, colors="g", linewidth=2.0, label='Flow 0')
ax2.vlines(ps.arrivals[1], 0.0, 0.7, colors="r", linewidth=2.0, label='Flow 1')
ax2.set_title("Departure times from the WFQ server")
ax2.set_xlabel("time")
ax2.set_ylim([0, 1.5])
ax2.set_xlim([0, max(ps.arrivals[0]) + 10])
ax2.grid(True)
ax2.legend()

fig.savefig("wfq.png")

plt.show()
