from typing import Callable, Iterable, Union
import numpy as np
import matplotlib.pyplot as plt

class experimental_unit:
    
    def __init__(self,
        problem_solver: Callable, 
        noise: Callable, 
        parameters: Iterable, 
        noise_level: float = 1,
        ):
        
        self.problem_solver_fun = problem_solver
        self.noise_fun = noise
        self.parameters = parameters
        self.noise_level = noise_level
        
    def problem_solver(self, x, y=None):
        # this method returns the dependent variable based on the independent variables x and y and the parameters
        if y is None:
            return self.problem_solver_fun(x, self.parameters)
        else:
            return self.problem_solver_fun(x, y, self.parameters)
    
    def noise(self, noise_level=None):
        # this method returns the noise
        if noise_level == None:
            noise_level = self.noise_level
            
        return self.noise_fun(noise_level)
    
    def step(self, x, y=None):
        # this method returns the observation which is the sum of the dependent variable and the noise
        return self.problem_solver(x, y) + self.noise()
    
    
def noise(noise_level: float) -> float:
    """
    This function returns a random number drawn from a normal distribution with mean 0 and standard deviation noise_level
    
    Args:
        noise_level (float): the noise level
    
    Returns:
        float: a random number between
    """
    return np.random.normal(0, noise_level)

def linear_ground_truth(x: float, y: float=None, parameter: Union[list[float], float]=None) -> float:
    """
    This function returns the dependent variable based on the independent variables x and y and the parameters
    
    Args:
        x (float): the first independent variable
        y (float, Optional): the second independent variable
        parameter (list): the parameters for each independent variable; if y is None, then parameter must be a list of length 1; otherwise, parameter must be a list of length 2
    
    Returns:
        response (float): the response of the linear function
    """
    
    if parameter is not None and isinstance(parameter, (float, int)):
        parameter = [parameter]
            
    if y is not None:
        if parameter is None:
            parameter = [1, 1]
        
        assert parameter is not None and len(parameter) == 2, 'parameters must be a list of length 2'
        return parameter[0] * x + parameter[1] * y
    else:
        if parameter is None:
            parameter = [1]
        assert parameter is not None and len(parameter) == 1, 'parameters must be a list of length 1'
        return parameter[0] * x
    
def sigmoid_ground_truth(x: float, parameter: float=None, mirrored: bool=True) -> float:
    """
    This function returns the dependent variable based on the independent variables x and y and the parameters
    
    Args:
        x (float): the independent variable
        parameter (list): the parameter for the independent variable; Parameter must be a list of length 1
        mirrored (bool, optional): If True, then the sigmoid function is mirrored horizontally. Defaults to True.
        
    Returns:
        response (float): the response of the sigmoid function 
    """
    if parameter is None:
        parameter = 0
        
    assert isinstance(parameter, (float, int)), 'parameter must be a float or an int'
    response = 1 / (1 + np.exp(-x + parameter))
    if mirrored:
        return 1 - response
    else:
        return response
    
    
class DriftDiffusionModel:
    def __init__(self, drift_rate, starting_point, threshold, noise_std, time_step=0.01):
        self.drift_rate = drift_rate
        self.starting_point = starting_point
        self.threshold = threshold
        self.noise_std = noise_std
        self.time_step = time_step

    def run_simulation(self, collect_trace=False):
        evidence = self.starting_point
        time = 0
        evidence_trace = []

        while abs(evidence) < self.threshold:
            evidence += self.drift_rate * self.time_step + np.random.normal(0, self.noise_std) * np.sqrt(self.time_step)
            time += self.time_step
            if collect_trace:
                evidence_trace.append(evidence)

        return np.sign(evidence), time, evidence_trace if collect_trace else None

    def plot_evidence_trace(self, num_simulations=10):
        traces = []

        for _ in range(num_simulations):
            traces.append(self.run_simulation(collect_trace=True)[2])
            
        for trace in traces:
            plt.plot(np.arange(0, len(trace) * self.time_step, self.time_step), trace)
        plt.xlabel('Time')
        plt.ylabel('Evidence')
        plt.title('Evidence Trace')
        plt.axhline(y=self.threshold, color='r', linestyle='-')
        plt.axhline(y=-self.threshold, color='r', linestyle='-')
        plt.show()

    def plot_response_time_distribution(self, num_simulations=1000, bins=1001):
        response_times = []
        decisions = []

        for _ in range(num_simulations):
            decision, decision_time, _ = self.run_simulation()
            response_times.append(decision_time * decision)
            decisions.append(decision)

        plt.hist(response_times, bins=bins, color='blue', alpha=0.7)
        plt.xlabel('Response Time')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.show()
        
class MysteryDDM(DriftDiffusionModel):
    def __init__(self):
        super().__init__(drift_rate=0.1, starting_point=0, threshold=1, noise_std=0.5)
