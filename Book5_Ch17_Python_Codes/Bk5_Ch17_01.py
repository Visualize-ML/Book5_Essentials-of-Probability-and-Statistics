

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mean_array = []
num_dices  = 20 # n = 5, 10, 20
num_trials = 10000

# each trial: 10 dices and calculate mean
for i in np.arange(num_trials):
    
    sample_i = np.random.randint(low = 1, 
                                 high = 6 + 1, 
                                 size=(num_dices))
    
    mean_i   = sample_i.mean()
    mean_array.append(mean_i)

# plot the histogram of mean values at 50, 500, 5000 trials

for j in [100,1000,10000]: # m
    
    mean_array_j = mean_array[0:j]
    
    fig, ax = plt.subplots()
    
    sns.histplot(mean_array_j, kde = True,
                 stat="density",
                 binrange = [1,6],
                 binwidth = 0.2)
    
    mean_array_j = np.array(mean_array_j)
    
    mu_mean_array_j = mean_array_j.mean()
    
    ax.axvline(x = mu_mean_array_j,
               color = 'r',linestyle = '--')
    
    sigma_mean_array_j = mean_array_j.std() 
    
    ax.axvline(x = mu_mean_array_j + sigma_mean_array_j,
               color = 'r',linestyle = '--')
    
    ax.axvline(x = mu_mean_array_j - sigma_mean_array_j,
               color = 'r',linestyle = '--')
    
    plt.xlim(1,6)
    plt.ylim(0,1)
    plt.grid()
