

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
from matplotlib import cm

def generate_paths(step_num,path_num, p, up_down):
    
    np.random.seed(0)
    
    start_locs = np.zeros((1, path_num))

    # random walk
    step_shape = (step_num, path_num)

    steps = np.random.choice(a= up_down, size=step_shape,
                             p = p)
    paths = np.concatenate([start_locs, steps]).cumsum(0)

    return paths

up_down = [-1, 1]

p = 0.4 # probability of moving up
probs = [1-p, p]

step_num = 20 # n in binomial distribution

path_num = 20

colors = plt.cm.rainbow(np.linspace(0,1,path_num))

# generate random paths
paths = generate_paths(step_num, path_num, probs, up_down)

fig, ax = plt.subplots()

for i in np.arange(0,path_num):
    
    plt.plot(np.arange(step_num+1), paths[:,i], 
             marker='.', markersize=8, color = colors[i,:]);

plt.xlabel('Step')
plt.ylabel('Position')
plt.axis('scaled')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

ax.set_xlim(0,step_num)
plt.xticks(np.arange(0,step_num+1,5))

ax.set_ylim(np.floor(paths.min()) - 1,
            np.ceil(paths.max())  + 1)

plt.axhline(y=0, linestyle = '--', color = 'k')

#%% distribution of the finishing locations

fig, axes = plt.subplots(1, 3, figsize=(15,4), 
                               gridspec_kw={'width_ratios': [1, 1, 1]})

for i,path_num in enumerate([50, 100, 5000]):

    paths = generate_paths(step_num, path_num, probs, up_down)
    stop_locs = paths[-1,:]

    plt.sca(axes[i])
    plt.hist(stop_locs, bins = len(np.unique(stop_locs)), 
             density = True, edgecolor='grey',
             color= '#DBEEF3')

    plt.xlim(-step_num,step_num)
    plt.axvline(x=stop_locs.mean(), linestyle = '--', color = 'r')

    plt.xlabel('Finishing location')
    plt.ylabel('Probability')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim(0,0.12)
