import queue
from collections import namedtuple
import numpy as np
import pandas as pd

# This class is to model the system's logic, including inserting an event and generating the next event
class Environment():
    Event = namedtuple('Event', ['time', 'type', 'info'])
    def __init__(self): 
        self.eventQ = queue.PriorityQueue()
        self.now = 0

    def add_event(self, time, event_type, info):
        """ Insert an event into the environment.

        Args:
            time (float): the duration or the interval of the event
            event_type (str): the type of the event
            info (any): you can transfer some objects
        """        
        self.eventQ.put(self.Event(self.now + time, event_type, info))

    def next_event(self) -> Event:  
        assert not self.eventQ.empty()
        event = self.eventQ.get()
        self.now = event.time
        return event

# This class is to collect the information of a customer, including arrival time, service time and departure time. You can use these information to compute and then analyze the waiting time.
class Customer:
    def __init__(self, name, type_id, customer_id, arrival_time):
        self.name = name
        self.type_id = type_id
        self.customer_id = customer_id
        self.arrival_time = arrival_time
        
# This class is to generate a sequence of customers with the same distributions of inter-arrival times and service times. 
class CustomerGenerator:
    def __init__(self, env:Environment, name, type_id, interarrival_time, initial_delay = 0):
        self.env = env
        self.name = name
        self.type_id = type_id
        self.interarrival_time = interarrival_time
        self.customer_generated = 1
        self.env.add_event(initial_delay, 'Arrival', None)
        
    def generate(self):
        interarrival = self.interarrival_time()
        customer = Customer(self.name, self.type_id, self.customer_generated, self.env.now)
        # update information
        self.customer_generated += 1
        self.env.add_event(interarrival, 'Arrival', None)
        # print(customer.customer_id, customer.arrival_time)
        return customer

# This class is to model the server's logic, including receiving a customer, choosing a customer to serve, sending it out of the server.
class Server:
    def __init__(self, env:Environment, server_id, service_time, debug=False):
        self.env = env
        self.server_id = server_id
        self.service_time = service_time
        self.debug = debug
        self.Q = queue.Queue()
        # server state
        self.busy = False
        self.current_customer = None
        
    def receive(self, customer:Customer):
        self.Q.put(customer)
        if self.debug:
            print(f'Time: {self.env.now:.4f}\tEvent: Arrival\t{customer.name} {customer.customer_id} of type {customer.type_id} is comming.')
        # try to serve it if the server is idle
        if not self.busy:
            self.env.add_event(0, 'Start', None)
        
    def serve(self):
        assert not self.busy and not self.Q.empty()
        # serve a customer
        customer = self.Q.get()
        customer.start_time = self.env.now    
        customer.service_time = self.service_time()   
        if self.debug:
            print(f'Time: {self.env.now:.4f}\tEvent: Start\t{customer.name} {customer.customer_id} of type {customer.type_id} begins to be served.')
        # update state
        self.busy = True
        self.current_customer = customer
        self.env.add_event(customer.service_time, 'Depart', None)
        # print(customer.customer_id,"begin serve", customer.start_time)
        # print("waiting time ", customer.customer_id, customer.start_time - customer.arrival_time)
        return customer
        
        
    def depart(self) -> Customer:
        assert self.busy
        # send a customer
        customer = self.current_customer
        customer.depart_time = self.env.now
        if self.debug:
            print(f'Time: {self.env.now:.4f}\tEvent: Depart\t{customer.name} {customer.customer_id} of type {customer.type_id} is leaving.')
        # update state
        self.busy = False 
        self.current_customer = None 
        # try to serve it if the server is non-empty
        if not self.Q.empty():
            self.env.add_event(0, 'Start', None)
        return customer

def interarrival_time():
    return np.random.exponential(scale=1/(1.5), size=1)[0]

def service_time():
    return np.random.exponential(scale=1/2, size=1)[0]


def main(n):
    estimate = []
    while True:
        nums = None
        env = Environment()
        Generator = CustomerGenerator(env, 'Customer', 1, interarrival_time)
        Queue_server = Server(env, 0, service_time, debug=False)

        # control center
        while nums is None:
            event = env.next_event()
            if event.type == 'Arrival':
                customer = Generator.generate()
                Queue_server.receive(customer)
            elif event.type == 'Start':
                cus = Queue_server.serve()
                if cus.customer_id == n:
                    nums = cus.start_time - cus.arrival_time
            elif event.type == 'Depart':
                customer = Queue_server.depart()
            else:
                assert False, f'invalid type: {event.type}'

        # print(nums)
        estimate.append(nums)

        if (len(estimate) > 100): 
            break

    estimate_wat = np.mean(estimate)
    estimate_std = np.std(estimate)
    return [estimate_wat, estimate_wat-1.96*estimate_std/np.sqrt(len(estimate)), estimate_wat+1.96*estimate_std/np.sqrt(len(estimate))]



if __name__ == "__main__":
    # for i in [1, 10, 100, 1000, 10000]:
    #     result = main(i)
    #     print("n=" + str(i) + ":" + " Expectation: "+ str(result[0]) + " 95% CI: " + str(result[1:]))   
    np.random.seed(43)  
    print(main(10000)) #[1.490462614358172, 1.4525963926397583, 1.5283288360765859] when times = 10000