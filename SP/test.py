import numpy as np


lam = np.array([10,10,10,400,200,100, 100, 450])*1e6
size = np.array([100, 100, 200, 1400, 1400,1400, 1400,1400]) * 8
size = size

port_rate = 3e9
arrival_rate = ((lam/size))
theta = size / port_rate

mu = 1/theta

print(f'arrival rate (bit): {list(lam/1e6)} Mbit/s')
print(f"arrival rate (number): {list(arrival_rate/1e3)} thousand packets/s")
print(f"service rate: {list(mu/1e6)}  million packets/s")
print(f'packet size (byte): {size/8}')
print(f'If we choose: port rate {port_rate/1e9} Gbit/s')
print(f"then load: {sum(arrival_rate/mu)}")

print(arrival_rate/1e6, mu/1e6)
