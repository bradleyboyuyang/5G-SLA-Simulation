# Static Priority Simulation via Importance Sampling
Importance Sampling based on [ns.py](https://github.com/TL-System/ns.py) v0.3.1 (only for static priority).
- Here we assume the Poisson arrivals $\lambda_k$ for the flow k and exponential distributions $\mu_k$ for the flow k.

## Main Code
- `1_run_para.py`: Given a set of parameters (num_type, $\lambda$, $\mu$, threshold, flow_id), compute:
    - the mean and confidence interval of the probability of exceeding the threshold for the flow `flow_id`.
    - the weight of each event in causing the system to exceed the threshold for the first time.
    - save the result to the file `result/1_run_para.csv`.
- `2_unbias_check.py`: This is to check the unbiasness (run `1_run_para.py` for several times and see the probability of CI covering the theoretical value), and save the result of each epoch to the folder `result/2_unbias_check/`.
- `3_cross_entropy.py`: This is to run the cross-entropy method, and save the iterative result to the folder `result/3_cross_entropy/`.


## Core Code
Modify the `ns.py`, and wrap changes into the following files:
- `enviornment.py`:
    -   `current worklaod`: compute the lower bound response time for the current workload (including this packet) of the specific packet.
    - `check gamma`: record the reasons for exceeding the threshold for the first time
    -  `turning point`: decide whether to switch measure to the original one based on the lower bound time.
- `likelihood_ratio.py`:
    -  `logW`: return the log likelihood ratio, where the calculation can be divided into three cases (higher/same/lower priority).
    -  `cycle_logW`: sum up the log likelihood ratios of all the packets in the cycle. The result is the log likelihood ratio of this cycle. 
    -   `cycle_CE`: return a 4*K list recording the arrival_num, arrival_sum, service_num and service_sum of each type during the phi=1 period.
- `simulation.py`:
    -   `simulate_one_cycle`: return each type's adjusted number of packets, adjusted total response time, adjusted times of exceeding the threshold, as well as the reason for exceeding the threshold for the first time in this cycle.

Please see the pseudocode in the report or the source code for more details.