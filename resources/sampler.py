from typing import Optional, Union, List

import numpy as np
import pandas as pd
import itertools

from autora.variable import ValueType, VariableCollection


def random_sampler(
    variables: VariableCollection,
    num_samples: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
    sample_all: Optional[List[str]] = None,
):
    rng = np.random.default_rng(random_state)

    raw_conditions = {}
    for iv in variables.independent_variables:
        if iv.allowed_values is not None:
            if iv.name in sample_all:
                raw_conditions[iv.name] = iv.allowed_values
            else:
                raw_conditions[iv.name] = rng.choice(
                    iv.allowed_values, size=num_samples, replace=replace
                )
        elif (iv.value_range is not None) and (iv.type == ValueType.REAL):
            raw_conditions[iv.name] = rng.uniform(*iv.value_range, size=num_samples)

        else:
            raise ValueError(
                "allowed_values or [value_range and type==REAL] needs to be set for "
                "%s" % (iv)
            )
            
        # Handle variables specified in `sample_all`
    all_conditions = []
    for iv_name in sample_all:
        all_conditions.append(raw_conditions.pop(iv_name))
    
    # Create a Cartesian product of all `sample_all` variables
    sample_all_combinations = list(itertools.product(*all_conditions))
    
    # Combine with randomly sampled variables
    other_conditions = pd.DataFrame(raw_conditions)
    
    # Create the final dataframe
    final_rows = []
    for sample in sample_all_combinations:
        for _, row in other_conditions.iterrows():
            final_rows.append((*sample, *row.values))

    # Construct the final dataframe
    final_columns = sample_all + list(raw_conditions.keys())
    final_df = pd.DataFrame(final_rows, columns=final_columns)

    return final_df