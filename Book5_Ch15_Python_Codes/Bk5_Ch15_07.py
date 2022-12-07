
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt

mu_X1, sigma_X1 = -5, 3
mu_X2, sigma_X2 =  5, 4

rho = 0.5

MU =    [mu_X1, mu_X2]
SIGMA = [[sigma_X1**2, sigma_X1*sigma_X2*rho],
         [sigma_X1*sigma_X2*rho, sigma_X2**2]]

X_12 = np.random.multivariate_normal(MU, SIGMA, 5000)
X1 = X_12[:,0]
X2 = X_12[:,1]

fig, ax = plt.subplots()

plt.hist(X1,bins = 50, range = [-25,25], 
         density = True, facecolor="None",
         edgecolor = '#92CDDC',alpha = 0.5)

plt.axvline(x = mu_X1, color = 'b')
plt.axvline(x = mu_X1 - sigma_X1, color = 'b')
plt.axvline(x = mu_X1 + sigma_X1, color = 'b')

plt.hist(X2,bins = 50, range = [-25,25], 
         density = True, facecolor="None",
         edgecolor = '#92D050',alpha = 0.5)

plt.axvline(x = mu_X2, color = 'g')
plt.axvline(x = mu_X2 - sigma_X2, color = 'g')
plt.axvline(x = mu_X2 + sigma_X2, color = 'g')

plt.ylim(0,0.2)

#%% Y = X1 + X2

for rho in [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]:

    MU =    [mu_X1, mu_X2]
    SIGMA = [[sigma_X1**2, sigma_X1*sigma_X2*rho],
             [sigma_X1*sigma_X2*rho, sigma_X2**2]]

    X_12 = np.random.multivariate_normal(MU, SIGMA, 5000)
    X1 = X_12[:,0]
    X2 = X_12[:,1]
    
    Y = X1 + X2
    
    fig, ax = plt.subplots()

    plt.hist(X1,bins = 50, range = [-25,25], 
             density = True, facecolor="None",
             edgecolor = '#92CDDC')
    
    
    plt.hist(X2,bins = 50, range = [-25,25], 
             density = True, facecolor="None",
             edgecolor = '#92D050')

    plt.hist(Y,bins = 50, range = [-25,25], 
             density = True, color = '#FFB591',
             edgecolor = 'k',alpha = 0.5)

    plt.axvline(x = Y.mean(), color = 'r')
    plt.axvline(x = Y.mean() - Y.std(), color = 'r')
    plt.axvline(x = Y.mean() + Y.std(), color = 'r')
    plt.ylim(0,0.2)
