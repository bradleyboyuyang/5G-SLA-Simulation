import queue
from environment import Environment

# This class is to collect the information of a customer, including arrival time, service time and departure time. You can use these information to compute and then analyze the waiting time.
class Customer:
    def __init__(self, name, type_id, customer_id, arrival_time):
        self.name = name
        self.type_id = type_id
        self.customer_id = customer_id
        self.arrival_time = arrival_time
        
# This class is to generate a sequence of customers with the same distributions of inter-arrival times and service times. 
class CustomerGenerator:
    def __init__(self, env:Environment, name, type_id, arrival_dist, initial_delay = 0):
        self.env = env
        self.name = name
        self.type_id = type_id
        self.arrival_dist = arrival_dist
        self.customer_generated = 0
        self.env.add_event(initial_delay, 'Arrival', None)
        
    def generate(self):
        interarrival = self.arrival_dist()
        customer = Customer(self.name, self.type_id, self.customer_generated, self.env.now + interarrival)
        # update information
        self.customer_generated += 1
        self.env.add_event(interarrival, 'Arrival', None)
        return customer

# This class is to model the server's logic, including receiving a customer, choosing a customer to serve, sending it out of the server.
class Server:
    def __init__(self, env:Environment, server_id, service_dist, debug=False):
        self.env = env
        self.server_id = server_id
        self.service_dist = service_dist
        self.debug = debug
        self.Q = queue.Queue()
        # server state
        self.busy = False
        self.current_customer = None
        
    def receive(self, customer:Customer):
        self.Q.put(customer)
        if self.debug:
            print(f'Time: {self.env.now:.2f}\tEvent: Arrival\t{customer.name} {customer.customer_id} of type {customer.type_id} is comming.')
        # try to serve it if the server is idle
        if not self.busy:
            self.env.add_event(0, 'Start', None)
        
    def serve(self):
        assert not self.busy and not self.Q.empty()
        # serve a customer
        customer = self.Q.get()
        customer.start_time = self.env.now    
        customer.service_time = self.service_dist()   
        if self.debug:
            print(f'Time: {self.env.now:.2f}\tEvent: Start\t{customer.name} {customer.customer_id} of type {customer.type_id} begins to be served.')
        # update state
        self.busy = True
        self.current_customer = customer
        self.env.add_event(customer.service_time, 'Depart', None)
        
        
    def depart(self) -> Customer:
        assert self.busy
        # send a customer
        customer = self.current_customer
        customer.depart_time = self.env.now
        if self.debug:
            print(f'Time: {self.env.now:.2f}\tEvent: Depart\t{customer.name} {customer.customer_id} of type {customer.type_id} is served.')
        # update state
        self.busy = False 
        self.current_customer = None 
        # try to serve it if the server is non-empty
        if not self.Q.empty():
            self.env.add_event(0, 'Start', None)
        return customer