
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np

x_array = np.linspace(0, 4, 100)

dfn_array = [1, 2, 5, 20, 100]
dfd_array = [1, 2, 5, 20, 100]
dfn_array_, dfd_array_ = np.meshgrid(dfn_array, dfd_array)

#%% PDF of F Distributions

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

for dfn_idx, dfd_idx,ax in zip(dfn_array_.ravel(), dfd_array_.ravel(), axs.ravel()):


    title_idx = '$d_1$ = ' + str(dfn_idx) + '; $d_2$ = ' + str(dfd_idx)
    ax.plot(x_array, f.pdf(x_array, dfn_idx, dfd_idx),
            'b', lw=1)
    ax.set_title(title_idx)
    ax.set_xlim(0,4)
    ax.set_ylim(0,2)
    ax.set_xticks([0,1,2,3,4])
    ax.set_yticks([0,1,2])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

#%% CDF of F Distributions

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

for dfn_idx, dfd_idx,ax in zip(dfn_array_.ravel(), dfd_array_.ravel(), axs.ravel()):


    title_idx = '$d_1$ = ' + str(dfn_idx) + '; $d_2$ = ' + str(dfd_idx)
    ax.plot(x_array, f.cdf(x_array, dfn_idx, dfd_idx),
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