from typing import Callable, Iterable
import numpy as np

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
        
    def problem_solver(self, x, y):
        # this method returns the dependent variable based on the independent variables x and y and the parameters
        return self.problem_solver_fun(x, y, self.parameters)
    
    def noise(self, noise_level=None):
        # this method returns the noise
        if noise_level == None:
            noise_level = self.noise_level
            
        return self.noise_fun(noise_level)
    
    def step(self, x, y):
        # this method returns the observation which is the sum of the dependent variable and the noise
        return self.problem_solver(x, y) + self.noise()
    
    
def noise(noise_level):
    # this function returns a random number between -noise_level and noise_level
    return np.random.uniform(-noise_level, noise_level)

def linear_ground_truth(x, y=None, parameter=None):
    # this function returns the dependent variable based on the independent variables x and y and the parameters
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
    
def sigmoid_ground_truth(x, parameter=None, mirrored=True):
    if parameter is None:
        parameter = [0]
    assert parameter is not None and len(parameter) == 1, 'parameters must be a list of length 1'
    response = 1 / (1 + np.exp(-x + parameter[0]))
    if mirrored:
        return 1 - response
    else:
        return response