This is an example code for implementing the Wash Machine in [Event_based](Event_based.pdf).

# Introduction
This example is to realize the Wash Machine in [Event_based], which is actually a first-in-first-out single server queue. 

## Prerequisites
In order to understand the codes, you need to be familiar with the basic Python language, `namedtuple`, `class`, and some commonly used packages, including `queue`, etc.

## Basic Components
Here we have the following components, each of which is grouped in a `namedtuple` or `class`.

### Event
This object is the core of event-based simulation. Since it has two simple components, `time` and `type`, we use `namedtuple` to remember them, instead of `class`. 

### Environment
This class is to model the system's logic, including inserting an event and generating the next event.

### Customer
This class is to collect the information of a customer, including arrival time, service time and departure time. You can use these information to compute and then analyze the waiting time.

### CustomerGenerator
This class is to generate a sequence of customers with the same distributions of inter-arrival times and service times. 

### Server
This class is to model the server's logic, including receiving a customer, choosing a customer to serve, sending it out of the server. 