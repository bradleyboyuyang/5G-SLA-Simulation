### Generalization to K Classes

- `1_run_para_k.py`: only change two functions in the single py file, `main` and `get_flow_result`
- `2_unbias_check_k.py`: the same logic, sample code parameter is randomly chosen since optimal change of measure is unknown
- `3_cross_entropy_k.py`: the same logic

### Simulation Example
#### Settings
4 classes: event 1-4
`lam = [0.6, 0.2, 0.2, 0.2]`
`mu = [2, 1, 2, 2]`
`lam_tilt = [0.8333, 0.4666, 0.4666, 0.4666]`
`mu_tilt = [1.2, 0.2, 1.2, 1.2]`
`threshold = [[2, 3], [2, 3], [2, 3], [2, 3]]`

In a single run,the result is saved in `./result/1_run_para_1.csv`

In unbiasedness checking, take 50 iterations as an example, the result is saved in `./result/temp_t1_iter50_policy1.csv`

#### Some notes
- Sometimes the 95\% CI of expected waiting time does not cover true value
- The CI is very wide, maybe due to large relative error because of improper change of measure
- NAN values when number of cycles is small, may turn to a very small value when number of cycles is large



### Questions
- "mix_p": the prob for mixed measure method
- policy 1, policy 2, policy 3
- "pid"
- Why "threshold" needs to be increasingly inputted?
- Good change of measure for K classes, cross entropy? 




