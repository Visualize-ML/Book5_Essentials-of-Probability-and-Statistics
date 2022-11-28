
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

x_array = np.linspace(0,1,200)
alpha_array = [0.1, 0.5, 1, 2, 4]
beta_array = [0.1, 0.5, 1, 2, 4]
alpha_array_, beta_array_ = np.meshgrid(alpha_array, beta_array)

#%% PDF of Beta Distributions

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

for alpha_idx, beta_idx,ax in zip(alpha_array_.ravel(), beta_array_.ravel(), axs.ravel()):


    title_idx = '\u03B1 = ' + str(alpha_idx) + '; \u03B2 = ' + str(beta_idx)
    ax.plot(x_array, beta.pdf(x_array, alpha_idx, beta_idx),
            'b', lw=1)
    ax.set_title(title_idx)
    ax.set_xlim(0,1)
    ax.set_ylim(0,4)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,2,4])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

#%% CDF of Beta Distributions

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

for alpha_idx, beta_idx,ax in zip(alpha_array_.ravel(), beta_array_.ravel(), axs.ravel()):


    title_idx = '\u03B1 = ' + str(alpha_idx) + '; \u03B2 = ' + str(beta_idx)
    ax.plot(x_array, beta.cdf(x_array, alpha_idx, beta_idx),
            'b', lw=1)
    ax.set_title(title_idx)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')
        
