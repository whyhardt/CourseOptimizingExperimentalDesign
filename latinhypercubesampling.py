import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube


def latin_hypercube_sampling(input_ranges, n_samples):    
    n_variables = len(input_ranges)

    # Define the bounds for each variable
    bounds = np.array(input_ranges)

    # Generate QMC samples
    samples = LatinHypercube(d=n_variables, seed=42).random(n_samples)

    # Scale and shift the samples to match the specified input ranges
    samples = samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    return samples

def custom_latin_hypercube_sampling(input_ranges, n_samples):
    n_variables = len(input_ranges)
    lhs_matrix = np.zeros((n_samples, n_variables))

    # Generate random samples within each interval
    for i, (start, end) in enumerate(input_ranges):
        lhs_matrix[:, i] = np.random.uniform(start, end, n_samples)

    # Shuffle each column independently
    for i in range(n_variables):
        np.random.shuffle(lhs_matrix[:, i])

    return lhs_matrix

def random_sampling(input_ranges, n_samples, dist='uniform'):
    n_variables = len(input_ranges)
    random_matrix = np.zeros((n_samples, n_variables))

    # Generate completely random samples within each interval
    for i, (start, end) in enumerate(input_ranges):
        if dist == 'uniform':
            random_matrix[:, i] = np.random.uniform(start, end, n_samples)
        elif dist == 'normal':
            random_matrix[:, i] = np.random.normal(start, end, n_samples)

    return random_matrix

def plot_sampling(lhs_matrix, random_matrix, input_ranges, bins=5):
    # n_samples_lhs, n_variables_lhs = lhs_matrix.shape
    # n_samples_random, n_variables_random = random_matrix.shape

    # make a figure with two subplots for the two sampling methods
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # plot the LHS samples on the first subplot
    # the first value of each sample is the x-value and the second value is the y-value
    # also plot a 2DHist of the LHS samples
    axes[0].hist2d(lhs_matrix[:, 0], lhs_matrix[:, 1], bins=bins, cmap='Blues')
    axes[0].scatter(lhs_matrix[:, 0], lhs_matrix[:, 1], color='orange')
    axes[0].set_title('LHS Sampling')
    axes[0].set_xlabel('Factor 1')
    axes[0].set_ylabel('Factor 2')
    axes[0].set_xlim(input_ranges[0])
    axes[0].set_ylim(input_ranges[1])
    
    # plot the random samples on the second subplot
    # also plot a 2DHist of the random samples
    axes[1].hist2d(random_matrix[:, 0], random_matrix[:, 1], bins=bins, cmap='Blues')
    axes[1].scatter(random_matrix[:, 0], random_matrix[:, 1], color='orange')
    axes[1].set_title('Random Sampling')
    axes[1].set_xlabel('Factor 1')
    axes[1].set_ylabel('Factor 2')
    axes[1].set_xlim(input_ranges[0])
    axes[1].set_ylim(input_ranges[1])
    
    # count bins with no samples with numpy's histogram2d function
    n_empty_bins_lhs = 0
    n_empty_bins_random = 0
    lhs_hist2d = np.histogram2d(lhs_matrix[:, 0], lhs_matrix[:, 1], bins=bins)
    random_hist2d = np.histogram2d(random_matrix[:, 0], random_matrix[:, 1], bins=bins)
    lhs_hist2d[0][np.where(lhs_hist2d[0] == 0)] = -1
    lhs_hist2d[0][np.where(lhs_hist2d[0] > 0)] = 0
    n_empty_bins_lhs = -np.sum(lhs_hist2d[0])
    random_hist2d[0][np.where(random_hist2d[0] == 0)] = -1
    random_hist2d[0][np.where(random_hist2d[0] > 0)] = 0
    n_empty_bins_random = -np.sum(random_hist2d[0])
    
    # set title 
    fig.suptitle('LHS Sampling vs Random Sampling')
    axes[0].set_title(f'LHS Sampling ({n_empty_bins_lhs} empty bins)')
    axes[1].set_title(f'Random Sampling ({n_empty_bins_random} empty bins)')
    
    plt.show()
    

# Example usage
np.random.seed(42)
input_ranges = [(0, 1), (0, 1)]  # Specify the input ranges for each variable
n_samples = 100

lhs_matrix = latin_hypercube_sampling(input_ranges, n_samples)
random_matrix = random_sampling(input_ranges, n_samples, 'normal')
plot_sampling(lhs_matrix, random_matrix, input_ranges, bins=10)
