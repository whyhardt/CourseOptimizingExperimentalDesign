from typing import Callable, Iterable, Union, Optional
from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def cognitive_model(ratio, scatteredness, parameters=np.ones(2,)):
    """This cognitive model predicts response times in the 2AFC task
    
    Args:
        ratio (float): The balance of blue vs orange tiles (0 to 1)
        scatteredness (float): How randomly distributed the tiles are (0 to 1)
        parameters (Iterable[float], optional): Individual participant parameters. 
            parameters[0]: sensitivity to ratio
            parameters[1]: sensitivity to scatteredness

    Returns:
        float: Predicted response time
    """
    
    # This is a bell-shaped function that can saturate
    # Response time increases with ratio (more balanced = harder decision)
    # Response time also increases with scatteredness (more scattered = harder to process)
    response_time = (1 - np.exp(-np.power(ratio, 2) / parameters[0])) + np.power(scatteredness, parameters[1])
    
    return response_time


class experimental_unit:
    
    def __init__(self,
        cognitive_model: Callable, 
        parameters: Iterable, 
        noise_level: float = 1,
        ):
        
        self.cognitive_model_fun = cognitive_model
        self.parameters = parameters
        self.noise_level = noise_level
        
    def cognitive_model(self, ratio, scatteredness):
        # this method returns the dependent variable based on the independent variables x and y and the parameters
        return self.cognitive_model_fun(ratio, scatteredness, self.parameters)
    
    def noise(self, noise_level=None):
        # this method returns the noise
        if noise_level == None:
            noise_level = self.noise_level
            
        return self.noise_fun(noise_level)
    
    def step(self, ratio, scatteredness, noise=True):
        # this method returns the observation which is the sum of the dependent variable and the noise
        if noise:
            obs =  self.cognitive_model(ratio, scatteredness) + np.random.normal(0, self.noise_level)
        else:
            obs = self.cognitive_model(ratio, scatteredness)
        
        # make obs an array if obs is a scalar
        if not isinstance(obs, Iterable):
            obs = np.array(obs)
            
        if len(obs.shape) == 0:
            obs = obs.reshape(-1)
            
        return np.max(np.stack((np.zeros_like(obs), obs), axis=1), axis=1)


def generate_dataset(experimental_units: Iterable[experimental_unit], conditions: Iterable[Iterable[Number]], n_repetitions: int = 1, train_ratio: float = 1.0):
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
                observation = experimental_units[i].step(conditions[:, 0], conditions[:, 1])
                dataset[i, :, k, 0] += i
                dataset[i, :, k, 1:1+conditions.shape[-1]] = conditions
                dataset[i, :, k, -1] = observation
    
    dataset_train = dataset[:, :int(dataset.shape[1] * train_ratio)].reshape(-1, dataset.shape[-1])
    if train_ratio < 1.0:
        dataset_test = dataset[:, int(dataset.shape[1] * train_ratio):].reshape(-1, dataset.shape[-1])
    else:
        dataset_test = None
        
    # if shuffle:
    #     np.random.shuffle(dataset)
        
    return dataset_train, dataset_test


def twoafc(
    parameters: np.ndarray,
    name="2AFC",
    resolution=100,
    minimum_value_condition: float = 0.,
    maximum_value_condition: float = 1.,
    discrete_iv: bool = False,
):
    """
    2AFC experiment with two independent variables

    Args:
        parameters: abstract participant parameters for the underlying ground truth model; must be an array of shape (n_units, 2)
        name: name of the experiment
        resolution: number of allowed values for stimulus
        Examples:
        >>> s = twoafc()
        >>> s.run(np.array([[.2,.1]]), random_state=42)
            participant id  ratio   scatteredness   response time
        0   0               1.0     1.0             1.592393
    """
    
    if len(parameters.shape) == 1:
        parameters = parameters.reshape(1, 2)
    
    # check that parameters are positive
    if any(parameters[:, 0] < 0):
        raise ValueError("The given parameters must be in the range [0, inf).")
    
    params = dict(
        name="2AFC",
        resolution=resolution,
        parameters=parameters,
    )
    
    participant_id = IV(
        name="participant_id",
        allowed_values=np.arange(
            0, len(parameters)
        ),
        # value_range=(0, len(parameters)-1),
        units="",
        variable_label="participant_id",
        type=ValueType.REAL,
    )
    
    if discrete_iv:
        kwargs = {
            'allowed_values': np.linspace(minimum_value_condition, maximum_value_condition, resolution),
        }
    else:
        kwargs = {
            'value_range': (minimum_value_condition, maximum_value_condition),
        }
    
    ratio = IV(
        name="ratio",
        units="",
        variable_label="ratio",
        type=ValueType.REAL,
        **kwargs,
    )

    scatteredness = IV(
        name="scatteredness",
        units="",
        variable_label="scatteredness",
        type=ValueType.REAL,
        **kwargs,
    )

    response_time = DV(
        name="response_time",
        value_range=(0, 100),
        units="seconds",
        variable_label="response_time",
        type=ValueType.REAL,
    )

    variables = VariableCollection(
        independent_variables=[participant_id, ratio, scatteredness],
        dependent_variables=[response_time],
    )

    unit_ids = np.arange(len(parameters))
    
    def run(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        added_noise: float = 0.01,
        random_state: Optional[int] = None,
    ):

        rng = np.random.default_rng(random_state)
        X = np.array(conditions)
        Y = np.zeros((X.shape[0], len(variables.dependent_variables)))
        
        for idx, x in enumerate(X):
            y = (cognitive_model(x[0], x[1], parameters[int(x[0])])).reshape(-1)
            if y == np.inf:
                print(f'smth wrong at index {idx}')
                print(f'conditions: {x}')
            Y[idx] = np.max(np.stack((np.zeros_like(y), y), axis=1), axis=1)
            
        experiment_data = pd.DataFrame(conditions)
        experiment_data.columns = [v.name for v in variables.independent_variables]
        experiment_data[variables.dependent_variables[0].name] = Y
        return experiment_data

    ground_truth = partial(run, added_noise=0.0)

    def domain():
        p_initial_values = variables.independent_variables[0].allowed_values
        trial_values = variables.independent_variables[1].allowed_values

        X = np.array(np.meshgrid(p_initial_values, trial_values)).T.reshape(-1, 2)
        return X

    def plotter(
        model=None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        p_id = np.array([0])

        x_limit = variables.independent_variables[1].value_range
        y_limit = variables.independent_variables[2].value_range
        x_label = "Ratio"
        y_label = "Scatteredness"
        
        # define the factor levels
        x = ratio.allowed_values if ratio.allowed_values is not None else np.linspace(*variables.independent_variables[1].value_range) 
        y = scatteredness.allowed_values if scatteredness.allowed_values is not None else np.linspace(*variables.independent_variables[2].value_range)
        x_mesh, y_mesh = np.meshgrid(x, y)
        p_id = np.full_like(x, p_id)
        sample_size = len(x)

        # collect the observations for each participant
        # add your code here (you can take the relevant code pieces from above):

        # collect the observations
        dvs = [dv.name for dv in variables.dependent_variables]

        # initiate the z array
        z_ground_truth = {dv: np.zeros((sample_size, sample_size)) for dv in dvs}
        if model is not None:
            z_model = {dv: np.zeros((sample_size, sample_size)) for dv in dvs}

        for i in range(sample_size):
            x = np.stack((p_id, x_mesh[i], y_mesh[i]), axis=-1)
            z = ground_truth(x)
            if model is not None:
                z_m = model.predict(x)
            for idx, dv in enumerate(dvs):
                z_ground_truth[dv][i, :] = z[dv]
                if model is not None:
                    z_model[dv][i, :] = z_m[idx]
        
        # make a surface plot to visualize the ground_truth
        for dv in dvs:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(x_mesh, y_mesh, z_ground_truth[dv], cmap=cm.Blues, alpha=0.7)
            if model is not None:
                ax.plot_surface(x_mesh, y_mesh, z_model[dv], cmap=cm.Reds, alpha=0.7)
            
            ax.set_xlim(x_limit)
            ax.set_ylim(y_limit)
            ax.set_xlabel(x_label, fontsize="large")
            ax.set_ylabel(y_label, fontsize="large")
            ax.set_zlabel(dv, fontsize="large")
            ax.set_title("2AFC; DV: " + dv, fontsize="x-large")
            plt.show()

    collection = SyntheticExperimentCollection(
        name=name,
        description=twoafc.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=twoafc,
    )
    
    return collection



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
    assert conditions.shape[-1]==2, "conditions must be an iterable of length 2."
    assert parameters.shape[-1]==2, "parameters must be an iterable of length 2."
    
    if len(conditions.shape) == 1:
        x, y = conditions[0], conditions[1]
    else:
        x, y = conditions[:, 0], conditions[:, 1]
    
    # this is an example of a bell-shaped function which can saturate
    dependent_variable = (1-np.exp(-np.power(x, 2)/parameters[0])) + np.power(y, parameters[1])

    return dependent_variable