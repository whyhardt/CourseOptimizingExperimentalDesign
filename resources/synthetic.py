from typing import Callable, Iterable, Union
import numpy as np
import matplotlib.pyplot as plt
import hssm


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
    
def sigmoid_ground_truth(x: float, parameter: Union[float, list[float]]=None) -> float:
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
        parameter = (5, 1)
        
    return 1 / (1 + np.exp(parameter[1]*(-x + parameter[0])))
    
def binomial_ground_truth(x: float, parameter: list[float]=None, response_time=False) -> float:
    """
    This function returns 1 or 0 based on the independent variable x and the parameters given to a sigmoidal function.
    
    Args:
        x (float): the independent variable
        parameter (list): the parameters for the sigmoidal function (0: bias, 1: sensitivity); Parameter must be a list of length 2
    
    Returns:
        response (float): the response of the working memory function 
    """
    if parameter is None:
        parameter = (0, 1)
    
    prob_wrong = sigmoid_ground_truth(x, parameter)
    response = np.random.choice((0, 1), p=np.array((prob_wrong, 1-prob_wrong)).reshape(-1,))
    
    if response_time:
        rt = np.random.lognormal(np.max((1, x-parameter[0])), 0.5)
        return response, rt
    else:
        return response
        
def multimodal_ground_truth(x: float, y:float, parameter: list[float, float, float, float, float]=None) -> float:
    """
    This function returns the dependent variable based on the independent variables x and y and the parameters
    
    Args:
        x (float): the first independent variable
        y (float): the second independent variable
        parameter (list): the parameters for each independent variable; Parameter must be a list of length 4
    
    Returns:
        response (float): the response of the multimodal function 
    """
    if parameter is None:
        parameter = [3, 1, 1, 2]
        
    assert isinstance(parameter, list) and len(parameter) == 4, 'parameters must be a list of length 4'
    wave = parameter[0]*np.sin(parameter[0] * x) + np.cos(parameter[1] * y)
    parabola = parameter[2]*x**2 + parameter[3]*y**2
    return wave + parabola
    raise NotImplementedError
    # return hssm.HSSM()
