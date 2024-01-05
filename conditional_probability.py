import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import seaborn as sns

from latinhypercubesampling import latin_hypercube_sampling, random_sampling

# Set random seed for reproducibility
np.random.seed(42)

# configure experiment
num_conditions = 5
num_samples_per_condition = 1000
noise_level = 1  # set added white noise level

# set conditions x1 and x2
conditions = latin_hypercube_sampling(((0,1), (0,1)), num_conditions)
x1 = conditions[:, 0]
x2 = conditions[:, 1]

# ground truth model
ground_truth = lambda x1, x2: 2 * x1 + .2 * x2 #+ 2 * x * y
noise = np.random.normal(0, 1, num_samples_per_condition)

# get observations based on the condition x
y = np.array([ground_truth(x1, x2) + noise_level * n for n in noise])

# Flatten the y array for scatter plot
y_flat = y.flatten()

# Scatter plot of y
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x1, x2, c=y, cmap='Blues')
plt.title('Scatter Plot of y')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

# Plot PDF of y based on one set of conditions
plt.subplot(1, 2, 2)
# plot y over the conditions with x1 and x2 being the axes and y being the color
plt.title('Scatter plot of y over the conditions')
plt.xlabel('y')
plt.ylabel('Probability Density')

# Add a fitted normal distribution to the histogram
mu, std = norm.fit(y[0])
# xmin, xmax = plt.xlim()
x = np.linspace(np.min(y[0]), np.max(y[0]), 100)
p = norm.pdf(x, mu, std)
sns.lineplot(p, color='orange', label='Fitted Normal Distribution')

plt.legend()
plt.tight_layout()
plt.show()