

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
from matplotlib import pyplot as plt

import scipy.stats as stats

beta_dist = stats.beta

alpha = 1 # degree of belief # 1, 2, 16

true_percentage = 0.45 # 0.3
Data_all_trials = stats.bernoulli.rvs(true_percentage, size=200)


#%% visualize data of trials

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 30))

trials_array = np.arange(0,200)+1
mask = (Data_all_trials == 1)

axs[0].plot(Data_all_trials[mask],trials_array[mask],
               color = 'b', linestyle = None,
               marker = '.', markersize = 20,
               label = 'Rabbit')

axs[0].plot(Data_all_trials[~mask],trials_array[~mask],
               color = 'r', linestyle = None,
               marker = '.', markersize = 20,
               label = 'Chicken')

axs[0].plot(Data_all_trials,trials_array,
               color = [0.8,0.8,0.8])

axs[0].set_ylim(1,200)
axs[0].set_xlim(-0.5,1.5)
axs[0].set_xticks([0,1])

axs[0].set_ylabel("Number of trials", rotation=90)
# axs[0].yaxis.tick_right()
# axs[0].yaxis.set_label_position("right")
axs[0].set_xlabel("Result of each trial")
# axs[0].invert_xaxis()
axs[0].legend()

# plt.setp(axs[0].get_xticklabels(), rotation=90, va="top", ha="center")
# plt.setp(axs[0].get_yticklabels(), rotation=90, va="center", ha="left")

ratio_rabbits = np.cumsum(Data_all_trials)/trials_array
ratio_chickens = 1 - ratio_rabbits

axs[1].plot(ratio_rabbits,trials_array,
            color = 'b', label = 'Rabbit')

axs[1].plot(ratio_chickens,trials_array,
            color = 'r', label = 'Chicken')

axs[1].set_ylim(1,200)
axs[1].set_xlim(0,1)
axs[1].set_xticks([0,0.5,1])

axs[1].set_ylabel("Number of trials", rotation=90)
# axs[1].yaxis.tick_right()
# axs[1].yaxis.set_label_position("right")
axs[1].set_xlabel("Ratio")
# axs[1].invert_xaxis()
axs[1].legend()

# plt.setp(axs[1].get_xticklabels(), rotation=90, va="top", ha="center")
# plt.setp(axs[1].get_yticklabels(), rotation=90, va="center", ha="left")


#%% Continuous variations of the posterior
# ridgeline style

from matplotlib.pyplot import cm
theta_array = np.linspace(0, 1, 500)

num_animals_array = np.arange(0,200 + 5,5)
num_animals_array = num_animals_array[::-1]
# reverse the sequence of layers

colors = cm.rainbow_r(np.linspace(0, 1, len(num_animals_array)))

fig, ax = plt.subplots(figsize=(8, 30))

for idx, num_animals_idx in enumerate(num_animals_array):
    
    height = num_animals_idx
    # random data generator
    data_idx = Data_all_trials[0:num_animals_idx]
    # actual percentage of rabbits is 30%

    num_rabbits_idx = data_idx.sum() # s
    posterior_pdf = beta_dist.pdf(theta_array, 
                      num_rabbits_idx + alpha,  # s + alpha
                      num_animals_idx - num_rabbits_idx + alpha) # n - s + alpha
    
    ratio = 1.2
    ax.plot(theta_array, 
            posterior_pdf * ratio + height, 
            color = [0.6,0.6,0.6])
    
    ax.fill_between(theta_array, height, 
                    posterior_pdf * ratio + height, 
                    color=colors[idx])

ax.set_xlim(0,1)
ax.set_xlabel('Posterior')
ax.set_ylabel('Number of trials')


#%% snapshots of posterior curves
# locations of snapshots

num_animals_array = [0, 1, 2, 3, 4, 5, 10, 100, 200]

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

for ax_idx, num_animals_idx in zip(axs.ravel(), num_animals_array):
    
    # random data generator
    data_idx = Data_all_trials[0:num_animals_idx]
    # actual percentage of rabbits is 30%

    num_rabbits_idx = data_idx.sum() # s
    posterior_pdf = beta_dist.pdf(theta_array, 
                      num_rabbits_idx + alpha,  # s + alpha
                      num_animals_idx - num_rabbits_idx + alpha) # n - s + alpha
    
    loc_max = theta_array[np.argmax(posterior_pdf)]
    # location of MAP
    
    ax_idx.plot(theta_array, posterior_pdf)
    ax_idx.axvline(x = loc_max, color = 'r', linestyle = '--')
    ax_idx.set_title("Number of animals: %d; number of rabbits: %d" 
              % (num_animals_idx, num_rabbits_idx))
    
    ax_idx.set_xlabel('Percentage of rabbits, $\u03B8$')
    ax_idx.fill_between(theta_array, 0, posterior_pdf, color="#DEEAF6")
    ax_idx.axvline(x = true_percentage, color = 'k', linestyle = '--')
    ax_idx.set_xlim(0,1)
    ax_idx.set_yticks([0,5,10,15])
    ax_idx.set_ylim(0,15)

