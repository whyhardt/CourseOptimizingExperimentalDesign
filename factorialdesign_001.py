# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data for a 2x2 factorial design
np.random.seed(42)
levels = [5,2]  # Two levels for each factor
factors = len(levels)
replicates = 5
total_runs = replicates * np.prod(levels)
responses = []
run_combinations = []

# ground truth model
if len(levels) == 1:
    ground_truth = lambda x: 2 * x
elif len(levels) == 2:
    ground_truth = lambda x, y: 2 * x + .2 * y #+ 2 * x * y

# Generate factorial combinations
combinations = np.array(np.meshgrid(*[range(1, level + 1) for level in levels])).T.reshape(-1, factors)

# create response data
for run in range(replicates):
    response = np.array([ground_truth(*combination) for combination in combinations]) + np.random.normal(0, 1, combinations.shape[0])
    responses.append(response)
    run_combinations.append(combinations)
responses = np.concatenate(responses)
run_combinations = np.concatenate(run_combinations)

# Create a DataFrame to store the experimental data
data = pd.DataFrame(run_combinations, columns=[f'Factor {i+1}' for i in range(factors)])
data['Response'] = responses

# Display the generated data
print("Generated Experimental Data:")
print(data)

# Plot the interaction effect using seaborn
if len(levels) == 1:
    sns.lmplot(x='Factor 1', y='Response', data=data, ci=None)
    plt.title('Main Effect in a 1x2 Factorial Design')
    plt.show()
elif len(levels) == 2:
    sns.lmplot(x='Factor 1', y='Response', hue='Factor 2', data=data, ci=None)
    plt.title('Interaction Effect in a 2x2 Factorial Design')
    plt.show()

# Perform a two-way ANOVA using statsmodels
# formula = 'Response ~ C(`Factor 1`) * C(`Factor 2`)'
# model = ols(formula, data=data).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)

# # Display the ANOVA table
# print("\nANOVA Table:")
# print(anova_table)
