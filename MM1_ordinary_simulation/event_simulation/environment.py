import queue
from collections import namedtuple

class Environment():
    Event = namedtuple('Event', ['time', 'type', 'info'])
    def __init__(self):
        """
            This class is to collect events and generate the next event.
        """        
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
        """ Generate the next event.

        Returns:
            Event: includes time and type.
        """        
        assert not self.eventQ.empty()
        event = self.eventQ.get()
        self.now = event.time
        return event
    
if __name__ == "__main__":    
    # An example of priority queue.
    Event = namedtuple('Event', ['time', 'type', None])
    q = queue.PriorityQueue()
    q.put(Event(1, 'ace'))
    q.put(Event(40, 333))
    q.put(Event(3, 'afd'))
    q.put(Event(5, '4asdg'))
    # 1 has the highest priority
    while not q.empty():
        print(q.get())


