"""AutoRA components for the 2AFC task: variables, theorist, experimentalists, experiment runners, loop.

The closed loop works on an AutoRA `StandardState`:
    state.variables         what can be varied (ratio, scatteredness) and what is measured (choice_blue, rt_ms)
    state.conditions        the trials the experimentalist proposes next
    state.experiment_data   all trials played so far (one row per trial)
    state.models            the theorist's fitted models (the last one is the newest)

Every component is a function on the state (AutoRA's `on_state`):
    theorist(state)                                    fits the GP to experiment_data, appends it to models
    experimentalist(state, num_samples=10, ...)        writes the next trials to conditions
    experiment_runner(state)                           plays conditions, appends the answers to experiment_data

Students do not need AutoRA: experiments/twoafc.py --experimentalist file.py:function loads a plain
sampler written as in Tutorial 06 and wraps it (wrap_sampler):

    def my_sampler(gp, n, rng):             # or my_sampler(gp, data, n, rng)
        ...                                 # gp: the GP fitted to all trials so far (resources/gp.py:
                                            #     latent(), p_blue(), rt_prediction(), information())
        return next_trials                  # DataFrame with n rows and the columns ratio, scatteredness
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from autora.state import Delta, StandardState, estimator_on_state, experiment_runner_on_state, on_state
from autora.variable import DV, IV, VariableCollection
from sklearn.base import BaseEstimator

from resources.gp import GroupGP
from resources.twoafc_tools import RATIO_RANGE, SimulatedGroup, candidate_grid

COLUMNS = ["ratio", "scatteredness", "choice_blue", "rt_ms"]

VARIABLES = VariableCollection(
    independent_variables=[IV(name="ratio", value_range=RATIO_RANGE), IV(name="scatteredness", value_range=(0.0, 1.0))],
    dependent_variables=[DV(name="choice_blue"), DV(name="rt_ms")],
)


def new_state() -> StandardState:
    """An empty state: the 2AFC variables, no conditions, no data, no models."""
    return StandardState(variables=VARIABLES, experiment_data=pd.DataFrame(columns=COLUMNS))


def add_data(state: StandardState, trials: pd.DataFrame) -> StandardState:
    """Append played trials (columns ratio, scatteredness, choice_blue, rt_ms) to the state's experiment_data."""
    new = trials[COLUMNS].copy()
    new["choice_blue"] = new["choice_blue"].fillna(-1).astype(int)        # -1 = no answer in time
    return state + Delta(experiment_data=new)


# ------------------------------------------------------------------ theorist

class GPTheorist(BaseEstimator):
    """The course theorist as a scikit-learn estimator (what AutoRA expects): a GroupGP on choice and RT.

    fit(X, y): X = ratio, scatteredness; y = choice_blue, rt_ms.  predict(X): P(blue) and the typical RT (ms).
    The fitted GroupGP is available as .gp_ for experimentalists.
    """

    def fit(self, X, y):
        d = pd.concat([pd.DataFrame(np.asarray(X, float), columns=["ratio", "scatteredness"]),
                       pd.DataFrame(np.asarray(y, float), columns=["choice_blue", "rt_ms"])], axis=1)
        d["choice_blue"] = d["choice_blue"].fillna(-1).astype(int)
        self.gp_ = GroupGP().fit(d)
        return self

    def predict(self, X):
        t = pd.DataFrame(np.asarray(X, float), columns=["ratio", "scatteredness"])
        p, _ = self.gp_.p_blue(t)
        rt, _, _ = self.gp_.rt_prediction(t)
        return np.c_[p, np.exp(rt)]


theorist = estimator_on_state(GPTheorist())


# ------------------------------------------------------------------ experimentalists

@on_state(output=["conditions"])
def random_experimentalist(variables, num_samples: int = 10, random_state: Optional[int] = None):
    """The random sampler: ratio uniform in its range (0.1 to 0.9), scatteredness uniform in [0, 1]."""
    rng = np.random.default_rng(random_state)
    ranges = {v.name: v.value_range for v in variables.independent_variables}
    return pd.DataFrame({k: rng.uniform(*ranges[k], num_samples) for k in ["ratio", "scatteredness"]})


@on_state(output=["conditions"])
def uncertainty_experimentalist(models, num_samples: int = 10, random_state: Optional[int] = None):
    """GP uncertainty sampling (Session 6, Tutorial 06): the candidates with the largest expected gain, choice + RT."""
    rng = np.random.default_rng(random_state)
    candidates = candidate_grid()
    gain = models[-1].gp_.information(candidates, rng=rng)["total"].to_numpy()
    best = np.argsort(-(gain + 1e-9 * rng.random(len(gain))))[:num_samples]
    return candidates.iloc[best].reset_index(drop=True)


EXPERIMENTALISTS = {"random": random_experimentalist, "uncertainty": uncertainty_experimentalist}


def wrap_sampler(sampler: Callable) -> Callable:
    """Turn a plain sampler as in Tutorial 06, sampler(gp, n, rng) or sampler(gp, data, n, rng), into an
    AutoRA experimentalist. gp: the GP fitted to all trials so far; data: those trials; returns the next n
    trials (DataFrame with ratio, scatteredness)."""
    import inspect
    takes_data = "data" in inspect.signature(sampler).parameters

    @on_state(output=["conditions"])
    def experimentalist(models, experiment_data, num_samples: int = 10, random_state: Optional[int] = None):
        rng = np.random.default_rng(random_state)
        gp = models[-1].gp_
        if takes_data:
            return sampler(gp=gp, data=experiment_data, n=num_samples, rng=rng)
        return sampler(gp, num_samples, rng)

    return experimentalist


def load_experimentalist(spec: str) -> Callable:
    """'random', 'uncertainty', or 'path/to/file.py:function_name'.

    The function in the file is your sampler, written as in Tutorial 06: sampler(gp, n, rng), or
    sampler(gp, data, n, rng) if it also needs the trials so far. It must return a DataFrame with n
    rows and the columns ratio and scatteredness. (A function already decorated with AutoRA's
    @on_state is used as it is.)
    """
    if spec in EXPERIMENTALISTS:
        return EXPERIMENTALISTS[spec]
    import importlib.util
    path, _, name = spec.partition(":")
    module_spec = importlib.util.spec_from_file_location("student_sampler", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    function = getattr(module, name or "sampler")
    import inspect
    if "gp" in inspect.signature(function).parameters:
        return wrap_sampler(function)
    return function


# ------------------------------------------------------------------ experiment runners

def simulated_runner(group: Optional[SimulatedGroup] = None, seed: Optional[int] = None) -> Callable:
    """Experiment runner on the state: a simulated group plays the conditions."""
    group = group if group is not None else SimulatedGroup(seed=seed)
    return experiment_runner_on_state(lambda conditions: group.run(conditions)[COLUMNS])


# ------------------------------------------------------------------ the loop

def closed_loop(experimentalist: Callable, runner: Callable, n_trials: int = 100, n_start: int = 20,
                cycle: int = 10, random_state: Optional[int] = None, state: Optional[StandardState] = None):
    """Run the loop: n_start random trials, then cycles of theorist -> experimentalist -> experiment runner."""
    rng = np.random.default_rng(random_state)
    state = state if state is not None else new_state()
    state = random_experimentalist(state, num_samples=n_start, random_state=int(rng.integers(2**31)))
    state = runner(state)
    while len(state.experiment_data) < n_trials:
        n = min(cycle, n_trials - len(state.experiment_data))
        state = theorist(state)
        state = experimentalist(state, num_samples=n, random_state=int(rng.integers(2**31)))
        state = runner(state)
    return state
