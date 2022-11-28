

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 1024 + 1
digits = str(mp.pi)[2:]
len(digits)

digits_list = [int(x) for x in digits]

digits_array  = np.array(digits_list)
digits_matrix = digits_array.reshape((32, 32))

# different color
# distribution at different steps

# make a heatmap
import seaborn as sns

fig, ax = plt.subplots()

ax = sns.heatmap(digits_matrix, vmin=0, vmax=9,
                 cmap="RdYlBu_r", 
                 yticklabels=False,
                 xticklabels=False)

ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False)

num_digits_array = [100,1000,10000,100000,1000000]

for num_digits in num_digits_array:
    
    mp.dps = num_digits + 1
    digits = str(mp.pi)[2:]
    len(digits)
    
    digits_list = [int(x) for x in digits]
    
    digits_array  = np.array(digits_list)
    
    fig, ax = plt.subplots()

    counts = np.bincount(digits_array)
    
    ax.barh(range(10), counts, align='center', edgecolor = [0.8,0.8,0.8])
    
    for i, v in enumerate(counts):
        ax.text(v + num_digits/400, i, str(v), color='k', va='center')
    
    ax.axvline(x = num_digits/10, color = 'r', linestyle = '--')
    plt.yticks(range(10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Digit')
