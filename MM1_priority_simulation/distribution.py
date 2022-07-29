import numpy as np

class Distribution:
    def __init__(self, offset:float):
        self.offset = offset

    def sample(self) -> float:
        raise NotImplementedError

    def __call__(self):
        return self.sample()

    def input(self, x:float) -> float:
        return x - self.offset

    def output(self, x:float) -> float:
        return x + self.offset

    def logW(self, x:float) -> float:
        raise NotImplementedError

    def logW_tail(self, x:float) -> float:
        raise NotImplementedError

class ExponDistri(Distribution):
    def __init__(self, rates:list[float], seed:int=None, eps:float=1e-7):
        """An exponential random variable generator and compute log-likelihood ratio. 
        However, due to precision limitations of simpy, we cannot sample too small value.

        Args:
            rates (list[float]): rates
            eps (float, optional): offset. Defaults to 1e-7.
            seed (int, optional): random seed. Defaults to None.
        """        
        super().__init__(eps)
        self.rates = np.array(rates)
        self.num_phase = len(self.rates)
        self.seed = seed
        self.random = np.random.RandomState(seed=seed) 
        
    def sample(self, phase:int=0) -> float:
        """Sample an exponential random variable by using the rate `rates[phase]`. 

        Args:
            phase (int or narray, optional): phase to sample. Defaults to 0.

        Returns:
            float: an exponential random variable 
        """        
        sample = self.random.exponential(scale=1/self.rates[phase])
        return self.output(sample)
    
    def _logpdf(self, x:float, phase:int=0) -> float:
        lam = self.rates[phase]
        return np.log(lam) - lam * x

    def _logtail(self, x:float, phase:int=0) -> float:
        return -self.rates[phase] * x

    def logW(self, x:float, phase:int) -> float:
        """log-likelihood ratio of x sampling from phase, w.r.t. phase 0

        Args:
            x (float): input
            phase (int | list[int], optional): phase that x samples from. Defaults to 0.

        Returns:
            float|list[float]: log-likelihood ratio
        """        
        if phase == 0:
            return 0
        else:
            x = self.input(x)
            return self._logpdf(x, 0) - self._logpdf(x, phase)

    def logW_tail(self, x:float, phase:int) -> float:
        """log-tail ratio of x sampling from phase, w.r.t. phase 0

        Args:
            x (float): input
            phase (int | list[int], optional): phase that x samples from. Defaults to 0.

        Returns:
            float|list[float]: log-tail ratio
        """    
        if phase == 0:
            return 0
        else:
            x = self.input(x)
            return self._logtail(x, 0) - self._logtail(x, phase)

if __name__ == '__main__':
    rates = [0.01, 0.1, 1, 10, 100, 1000]
    exp = ExponDistri(rates)
    print(exp.sample([2,3,4]))
    print(exp.logW(3, [2,3,4,5]))