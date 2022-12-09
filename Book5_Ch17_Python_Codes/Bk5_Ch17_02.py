

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# create a population
num_population = 100000
X1 = np.random.normal(loc=-5, scale=1.0, size=int(num_population/2))
X2 = np.random.normal(loc=5,  scale=3,   size=int(num_population/2))

X = np.concatenate((X1, X2), axis=None)

fig, ax = plt.subplots()

sns.kdeplot(X, fill = True)

mu_X = X.mean()

ax.axvline(x = mu_X,
           color = 'r',linestyle = '--')

sigma_X = X.std() 

ax.axvline(x = mu_X + sigma_X,
           color = 'r',linestyle = '--')

ax.axvline(x = mu_X - sigma_X,
           color = 'r',linestyle = '--')

plt.grid()

#%%

num_draws  = 10
num_trials = 5000

mean_array = []

# each trial: 10 dices and calculate mean
for i in np.arange(num_trials):
    
    indice_i = np.random.randint(low = 0, 
                                 high = num_population, 
                                 size=(num_draws))
    sample_i = X[indice_i]
    
    mean_i   = sample_i.mean()
    mean_array.append(mean_i)

# plot the histogram of mean values at 50, 500, 5000 trials

for j in [50,500,5000]: # m
    
    mean_array_j = mean_array[0:j]
    
    fig, ax = plt.subplots()
    
    sns.histplot(mean_array_j, kde = True,
                 stat="density",
                 binrange = [-10,10],
                 binwidth = 0.5)
    
    mean_array_j = np.array(mean_array_j)
    
    mu_mean_array_j = mean_array_j.mean()
    
    ax.axvline(x = mu_mean_array_j,
               color = 'r',linestyle = '--')
    
    sigma_mean_array_j = mean_array_j.std() 
    
    ax.axvline(x = mu_mean_array_j + sigma_mean_array_j,
               color = 'r',linestyle = '--')
    
    ax.axvline(x = mu_mean_array_j - sigma_mean_array_j,
               color = 'r',linestyle = '--')
    
    plt.xlim(-10,10)
    plt.ylim(0,0.3)
    plt.grid()


#%% distributions of mean of mean

num_trials = 5000

fig, ax = plt.subplots()

# each trial: 10 dices and calculate mean
for num_draws in [4,8,16]:
    
    mean_array = []
    
    for i in np.arange(num_trials):

        indice_i = np.random.randint(low = 0, 
                                     high = num_population, 
                                     size=(num_draws))
        sample_i = X[indice_i]
        
        mean_i   = sample_i.mean()
        mean_array.append(mean_i)
        
        # finishing the generation of mean array

    sns.kdeplot(mean_array, fill = True)

plt.xlim(-10,10)
plt.ylim(0,0.3)
plt.grid()

#%% SE: standard error

num_trials = 5000

SE_array = []

n_array = np.linspace(4,100,25)

for num_draws in n_array:
    
    mean_array = []
    
    for i in np.arange(num_trials):

        indice_i = np.random.randint(low = 0, 
                                     high = num_population, 
                                     size=(int(num_draws)))
        sample_i = X[indice_i]
        
        mean_i   = sample_i.mean()
        mean_array.append(mean_i)
        
        # finishing the generation of mean array

    mean_array = np.array(mean_array)
    SE_i = mean_array.std()
    
    SE_array.append(SE_i)

fig, ax = plt.subplots()

plt.plot(n_array,SE_array,
         marker = 'x', markersize = 12)
plt.xlim(4,100)
plt.ylim(0,3)
plt.grid()
