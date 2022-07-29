from Components import CustomerGenerator, Server
from environment import Environment

def arrival_dist():
    return 4

def service_dist():
    return 6

env = Environment()
CarGenerator = CustomerGenerator(env, 'Car', 0, arrival_dist)
Machine = Server(env, 0, service_dist, debug=True)

until = 100
# control center
while env.now < until:
    event = env.next_event()
    if event.type == 'Arrival':
        car = CarGenerator.generate()
        Machine.receive(car)
    elif event.type == 'Start':
        Machine.serve()
    elif event.type == 'Depart':
        car = Machine.depart()
    else:
        assert False, f'invalid type: {event.type}'