import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.qmc import LatinHypercube


n_samples = 100
lim = 1

# sample n points in a 2D space from a normal distribution
samples = np.random.uniform(low=-lim, high=lim, size=(n_samples, 2))

# # plot these samples
# plt.scatter(samples[:, 0], samples[:, 1])
# plt.xlim(-lim, lim)
# plt.ylim(-lim, lim)
# #turn of axes
# # plt.axis('off')
# plt.show()

# samples = np.random.normal(size=(n_samples, 2))

# # plot these samples
# plt.scatter(samples[:, 0], samples[:, 1])
# plt.xlim(-lim, lim)
# plt.ylim(-lim, lim)
# plt.axis('off')
# plt.show()

samples = LatinHypercube(d=2, seed=42).random(n_samples)

# plot these samples
plt.scatter(samples[:, 0], samples[:, 1])
# plt.xlim(-lim, lim)
# plt.ylim(-lim, lim)
# plt.axis('off')
plt.major_ticks = np.linspace(0, 1, n_samples//2)
plt.grid()
plt.show()
