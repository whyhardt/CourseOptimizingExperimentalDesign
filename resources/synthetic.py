from typing import Callable, Iterable, Union
from numbers import Number
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
        
    def problem_solver(self, conditions):
        # this method returns the dependent variable based on the independent variables x and y and the parameters
        return self.problem_solver_fun(conditions, self.parameters)
    
    def noise(self, noise_level=None):
        # this method returns the noise
        if noise_level == None:
            noise_level = self.noise_level
            
        return self.noise_fun(noise_level)
    
    def step(self, conditions, noise=True):
        # this method returns the observation which is the sum of the dependent variable and the noise
        if noise:
            obs =  self.problem_solver(conditions) + self.noise()
        else:
            obs = self.problem_solver(conditions)
        
        # make obs an array if obs is a scalar
        if not isinstance(obs, Iterable):
            obs = np.array(obs)
            
        if len(obs.shape) == 0:
            obs = obs.reshape(-1)
            
        return np.max(np.stack((np.zeros_like(obs), obs), axis=1), axis=1)


def generate_dataset(experimental_units: Iterable[experimental_unit], conditions: Iterable[Iterable[Number]], n_repetitions: int, shuffle: bool = False):
    # check that each element of conditions has only two numeric sub-elements
    assert any([[isinstance(e, Number) for e in c] for c in conditions]), 'Some elements in the conditions argument appear to be not of length 2 or are non-numeric'
    
    n_experimental_units = len(experimental_units)
    n_conditions = len(conditions)
    
    # create an array which will be the dataset
    dataset = np.zeros((n_experimental_units, n_conditions, n_repetitions, 2+conditions.shape[-1]))

    for i in range(n_experimental_units):
        # here we collect the observations for each experimental unit
        for k in range(n_repetitions):
                # here we collect the observations for each repetition for each condition for each experimental unit
                # dataset[i, :, k] = experimental_units[i].step(conditions[:, 0], conditions[:, 1])
                observation = experimental_units[i].step(conditions)
                dataset[i, :, k, 0] += i
                dataset[i, :, k, 1:1+conditions.shape[-1]] = conditions
                dataset[i, :, k, -1] = observation
    
    if shuffle:
        np.random.shuffle(dataset)
    
    dataset_flat = dataset.reshape(-1, dataset.shape[-1])
    
    return dataset, dataset_flat


def noise(noise_level: float) -> float:
    """
    This function returns a random number drawn from a normal distribution with mean 0 and standard deviation noise_level
    
    Args:
        noise_level (float): the noise level
    
    Returns:
        float: a random number between
    """
    return np.random.normal(0, noise_level)


def linear_ground_truth(conditions, parameters: Union[list[float], float]=None) -> float:
    """
    This function returns the dependent variable based on the independent variables x and y and the parameters
    
    Args:
        x (float): the first independent variable
        y (float, Optional): the second independent variable
        parameter (list): the parameters for each independent variable; if y is None, then parameter must be a list of length 1; otherwise, parameter must be a list of length 2
    
    Returns:
        response (float): the response of the linear function
    """
    
    if isinstance(conditions, Iterable):
        assert len(conditions)<=2, "conditions must be an iterable of maximum length 2."
        if len(conditions) == 1:
            x = conditions[0]
        elif len(conditions) == 2:
            x, y = conditions[0], conditions[1]
        
        assert len(parameters)==len(conditions), "parameters must be an iterable of length 2."
        
    else:
        x, y = conditions, 0
        
    return parameters[0] * x + parameters[1] * y

    
def sigmoid_ground_truth(conditions, parameters: Union[float, list[float]]) -> float:
    """
    This function returns the dependent variable based on the independent variables x and y and the parameters
    
    Args:
        x (float): the independent variable
        parameter (list): the parameter for the independent variable; Parameter must be a list of length 1
        mirrored (bool, optional): If True, then the sigmoid function is mirrored horizontally. Defaults to True.
        
    Returns:
        response (float): the response of the sigmoid function 
    """
    
    if isinstance(conditions, Iterable):
        if len(conditions.shape) == 1:
            assert len(conditions)==1, "conditions must be an iterable of length 1."
            x = conditions[0]
        elif len(conditions.shape) == 2:
            assert conditions.shape[-1]==1, "conditions must be an iterable of shape (n, 1)."
            x = conditions[:, 0]
    else:
        x = conditions
    assert len(parameters)==2, "parameters must be an iterable of length 2."
    
    return 1 / (1 + np.exp(parameters[1]*(-x + parameters[0])))

    
def binomial_ground_truth(conditions, parameters: list[float], response_time=False) -> float:
    """
    This function returns 1 or 0 based on the independent variable x and the parameters given to a sigmoidal function.
    
    Args:
        x (float): the independent variable
        parameter (list): the parameters for the sigmoidal function (0: bias, 1: sensitivity); Parameter must be a list of length 2
    
    Returns:
        response (float): the response of the working memory function 
    """
    if isinstance(conditions, Iterable):
        assert len(conditions)==1, "conditions must be an iterable of length 1."
        x = conditions[0]
    else:
        x = conditions
    assert len(parameters)==2, "parameters must be an iterable of length 2."
    
    prob_wrong = sigmoid_ground_truth(x, parameters)
    response = np.random.choice((0, 1), p=np.array((prob_wrong, 1-prob_wrong)).reshape(-1,))
    
    if response_time:
        rt = np.random.lognormal(np.max((1, x-parameters[0])), 0.5)
        return response, rt
    else:
        return response

        
def wave_ground_truth(conditions, parameters: list[float, float, float, float, float]) -> float:
    """
    This function returns the dependent variable based on the independent variables x and y and the parameters
    
    Args:
        x (float): the first independent variable
        y (float): the second independent variable
        parameter (list): the parameters for each independent variable; Parameter must be a list of length 4
    
    Returns:
        response (float): the response of the multimodal function 
    """
    
    assert len(conditions)==2, "conditions must be an iterable of length 2."
    assert len(parameters)==4, "parameters must be an iterable of length 4."
    
    x, y = conditions[0], conditions[1]
        
    assert isinstance(parameters, list) and len(parameters) == 4, 'parameters must be a list of length 4'
    wave = parameters[0]*np.sin(parameters[0] * x) + np.cos(parameters[1] * y)
    parabola = parameters[2]*x**2 + parameters[3]*y**2
    return wave + parabola


def normal_ground_truth(conditions, parameters=np.ones(2,)):
    """This ground truth takes in two factors and a set of parameters and returns a response

    Args:
        x (float): The level of the first factor
        y (float): the level of the second factor
        parameters (Iterable[float], optional): The parameters give an individual configuration for each experimental unit. Defaults to np.ones(2,).

    Returns:
        float: Dependent variable which serves as the response
    """
    
    # this is an example of a ground truth function with two linear terms and a constant term 
    # dependent_variable = parameters[0] * x + parameters[1] * y + parameters[0] - parameters[1]
    assert len(conditions)==2, "conditions must be an iterable of length 2."
    assert len(parameters)==2, "parameters must be an iterable of length 2."
    
    x, y = conditions[0], conditions[1]
    
    # this is an example of a bell-shaped function which can saturate
    dependent_variable = (1-np.exp(-np.pow(x, 2)/parameters[0])) + np.pow(y, parameters[1])
    
    return dependent_variable