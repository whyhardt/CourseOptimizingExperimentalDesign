import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# this is going to be an example of a working memory task for remembering sequences of different lengths

# set up the experiment
conditions = np.arange(1, 15) 
bias = np.arange(5, 10)
# ground truth model is sigmoid function
def ground_truth(x, bias):
    return 1 - 1 / (1 + np.exp(-x + bias))

observations = np.zeros((len(bias), len(conditions)))

# run the experiment
for ib, b in enumerate(bias):
    for c in conditions:
        # get observation
        observations[ib, c-1] = ground_truth(c, b)
    
# compute mean for each condition
means = np.zeros(len(conditions))
for ic, c in enumerate(conditions):
    means[ic] = np.mean(observations[:, ic])
       
# compute variance for each condition
variances = np.zeros(len(conditions))
for ic, c in enumerate(conditions):
    variances[ic] = np.std(observations[:, ic])

print(np.array([conditions, means]))

# plot the observations
for ib, b in enumerate(bias):
    plt.plot(conditions, observations[ib], '-o', label=f'Participant {ib+1}')
plt.legend()
# plt.title('Working memory task: Remembering sequences of different lengths')
plt.xlabel('Sequence length')
plt.ylabel('Probability of remembering the sequence')
plt.show()

# plot the means and the variance above and beneath the mean and fill the area in between
# make the color of the variance dependending on the value of the variance
plt.plot(conditions, means, '-o', color='Orange', label='Mean')
# plt.fill_between(conditions, means - variances, means + variances, alpha=.2)
plt.fill_between(conditions, means, np.zeros_like(means), alpha=.2, color='Green')
plt.fill_between(conditions, means, np.ones_like(means), alpha=.2, color='Red')
plt.ylabel('Probability of remembering the sequence')
# make a second y axis
# ax2 = plt.twinx()
# ax2.set_ylabel('Variance')
# ax2.set_ylim(0, 0.3)
# plt.plot(conditions, variances, color='Blue', alpha=0.5, label='Variance')
plt.xlabel('Sequence length')
plt.legend()
# plt.title('Working memory task: Remembering sequences of different lengths')
plt.show()


# compute the conditional entropy
# H(X|Y) = - sum_x sum_y p(x,y) log p(x|y)
# X can be either 1 or 0
# means is the probability of X = 1
# Y is the sequence length

p_xy = observations
p_x = means
p_x_given_y = p_xy / p_x

H_x_given_y = - np.sum(p_xy * np.log(p_x_given_y), axis=0)
H_x_given_y = np.sum(H_x_given_y)
print(H_x_given_y)
