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
        # self.env.add_event(initial_delay, 'Arrival', type_id)
        
    def generate(self):
        interarrival = self.interarrival_time()
        customer = Customer(self.name, self.type_id, self.customer_generated, self.env.now)
        # update information
        self.customer_generated += 1
        self.env.add_event(interarrival, 'Arrival', self.type_id)
        return customer

# This class is to model the server's logic, including receiving a customer, choosing a customer to serve, sending it out of the server.
class Server:
    def __init__(self, env:Environment, server_id, service_time, debug=False):
        self.env = env
        self.server_id = server_id
        self.debug = debug
        self.service_time = service_time
        self.Q = queue.PriorityQueue()
        # server state
        self.busy = False
        self.current_customer = None
        
    def receive(self, customer:Customer):
        self.Q.put((customer.type_id, self.env.now, customer))
        if self.debug:
            print(f'Time: {self.env.now:.4f}\tEvent: Arrival\t{customer.name} {customer.customer_id} of type {customer.type_id} is comming.')
        # try to serve it if the server is idle
        if not self.busy:
            self.env.add_event(0, 'Start', None)
        
    def serve(self):
        assert not self.busy and not self.Q.empty()
        # serve a customer
        id, _, customer = self.Q.get()
        customer.start_time = self.env.now   
        customer.service_time = self.service_time(id)  
        if self.debug:
            print(f'Time: {self.env.now:.4f}\tEvent: Start\t{customer.name} {customer.customer_id} of type {customer.type_id} begins to be served.')
        # update state
        self.busy = True
        self.current_customer = customer
        self.env.add_event(customer.service_time, 'Depart', None)
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

def interarrival_time1():
    return np.random.exponential(scale=1/(0.6), size=1)[0]

def interarrival_time2():
    return np.random.exponential(scale=1/(0.2), size=1)[0]

def service_time(id):
    if id == 1:
        return np.random.exponential(scale=1/2, size=1)[0]
    else:
        return np.random.exponential(scale=1, size=1)[0]


def main(n):

    estimate1 = []
    estimate2 = []
    while True:
        nums1 = nums2 = None
        env = Environment()
        Queue_server = Server(env, 1, service_time, debug=False)

        Generator1 = CustomerGenerator(env, 'Customer', 1, interarrival_time1)
        Generator2 = CustomerGenerator(env, 'Customer', 2, interarrival_time2)

        rand = np.random.random()
        if rand > 0.75:
            Queue_server.receive(Generator2.generate())
            Generator1.generate()
        else:
            Queue_server.receive(Generator1.generate())
            Generator2.generate()

        # control center
        while ((nums1 == None) | (nums2 == None)) :
            event = env.next_event()
            if event.type == 'Arrival':
                if event.info == 1:
                    Queue_server.receive(Generator1.generate())
                else:
                    Queue_server.receive(Generator2.generate())

            elif event.type == 'Start':
                cus = Queue_server.serve()
                if cus.customer_id == n:
                    if cus.type_id == 1:
                        nums1 = cus.start_time - cus.arrival_time
                    else:
                        nums2 = cus.start_time - cus.arrival_time

            elif event.type == 'Depart':
                customer = Queue_server.depart()
            else:
                assert False, f'invalid type: {event.type}'
        
        estimate1.append(nums1)
        estimate2.append(nums2)
        if (len(estimate1) > 1000) & (len(estimate2) > 1000): 
            break

    estimate_wat1 = np.mean(estimate1)
    estimate_std1 = np.std(estimate1)
    estimate_wat2 = np.mean(estimate2)
    estimate_std2 = np.std(estimate2)
    return [estimate_wat1, estimate_wat1-1.96*estimate_std1/np.sqrt(n), estimate_wat1+1.96*estimate_std1/np.sqrt(n)],[estimate_wat2, estimate_wat2-1.96*estimate_std2/np.sqrt(n), estimate_wat2+1.96*estimate_std2/np.sqrt(n)]


# theoretical value: 0.5 and 1
if __name__ == "__main__":
    # for i in [1, 10, 100, 1000, 10000]:
    #     np.random.seed(43)
    #     result = main(i)
    #     print("n = " + str(i) + ":")
    #     print(" E[W1] estimate: "+ str(result[0][0]) + " 95% CI of E[W1]: " + str(result[0][1:]))
    #     print(" E[W2] estimate: "+ str(result[1][0]) + " 95% CI of E[W2]: " + str(result[1][1:]))  
    #     print("")
    #     print('=========================================================================================')   
    np.random.seed(43)  
    print(main(10000))