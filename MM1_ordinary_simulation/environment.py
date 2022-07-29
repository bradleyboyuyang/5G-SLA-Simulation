import queue
from collections import namedtuple

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
    

class Customer:
    def __init__(self, name, type_id, customer_id, arrival_time):
        self.name = name
        self.type_id = type_id
        self.customer_id = customer_id
        self.arrival_time = arrival_time

if __name__ == "__main__":    
    # An example of priority queue.
    Event = namedtuple('Event', ['time', 'type', 'info'])
    q = queue.PriorityQueue()
    # q.put(Event(1, 1,'ace'))
    # q.put(Event(40, 0,333))
    # q.put(Event(3, 0, 'afd'))
    # q.put(Event(5, 1, '4asdg'))
    cus1 = Customer("Tom", 1, 123, 0.5)
    cus2 = Customer("amm", 2, 124, 0.3)
    cus3 = Customer("rmm", 1, 125, 0.6)

    q.put((cus1.type_id, 125, cus1))
    q.put((cus2.type_id, 124, cus2))
    q.put((cus3.type_id, 123, cus3))

    # 1 has the highest priority
    while not q.empty():
        print(q.get())


