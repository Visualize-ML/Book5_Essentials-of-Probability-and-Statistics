

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
from scipy.stats import norm


X = np.arange(-3,3,0.05)
mu = 0
sigma = 1

#%% Standard normal PDF and evenly spaced x 

x_selected = np.linspace(-2.8, 2.8, num=29)

f_x = norm.pdf(X, loc=mu, scale=sigma)

fig, ax = plt.subplots()

plt.plot(X,f_x)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(x_selected)))

for i in np.linspace(0,len(x_selected)-1,len(x_selected)):
    
    x_selected_i = x_selected[int(i)]
    
    x_PDF = norm.pdf(x_selected_i)

    plt.vlines(x = x_selected_i,
               ymin = 0, ymax = x_PDF,
               color = colors[int(i)])
    
    plt.plot(x_selected_i, x_PDF, marker = 'x',color = colors[int(i)],
             markersize = 12)
    
ax.set_xlim(-3, 3)
ax.set_ylim(0,  0.5)
ax.set_xlabel('z')
ax.set_ylabel('PDF, $f_{Z}(z)$')

#%% Standard normal CDF and evenly spaced x 

F_x = norm.cdf(X, loc=mu, scale=sigma)

fig, ax = plt.subplots()

plt.plot(X,F_x)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(x_selected)))

for i in np.linspace(0,len(x_selected)-1,len(x_selected)):
    
    x_selected_i = x_selected[int(i)]
    
    x_CDF = norm.cdf(x_selected_i)

    plt.vlines(x = x_selected_i,
               ymin = 0, ymax = x_CDF,
               color = colors[int(i)])

    plt.hlines(y = x_CDF,
               xmin = -3, xmax = x_selected_i,
               color = colors[int(i)])
    
    plt.plot(x_selected_i, x_CDF, marker = 'x',color = colors[int(i)],
             markersize = 12)
    
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('PDF, $f_{Z}(z)$')

#%% Standard normal PDF and evenly spaced percentiles 

percentiles = np.linspace(0.025, 0.975, num=39)

f_x = norm.pdf(X, loc=mu, scale=sigma)

fig, ax = plt.subplots()

plt.plot(X,f_x)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(percentiles)))

for i in np.linspace(0,len(percentiles)-1,len(percentiles)):
    
    percentile = percentiles[int(i)]
    
    x_percent = norm.ppf(percentile)
    
    x_PDF = norm.pdf(x_percent, loc = mu, scale = sigma)
    
    plt.vlines(x = x_percent,
               ymin = 0, ymax = x_PDF,
               color = colors[int(i)])
    
    plt.plot(x_percent, x_PDF, marker = 'x',color = colors[int(i)],
             markersize = 12)
    
ax.set_xlim(-3, 3)
ax.set_ylim(0,  0.45)
ax.set_xlabel('z')
ax.set_ylabel('PDF, $f_{X}(z)$')

#%% Standard normal CDF and percentiles 

F_x = norm.cdf(X, loc=mu, scale=sigma)

fig, ax = plt.subplots()

plt.plot(X,F_x)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(percentiles)))

for i in np.linspace(0,len(percentiles)-1,len(percentiles)):
    
    percentile = percentiles[int(i)]
    
    x_percent = norm.ppf(percentile)
    plt.hlines(y = percentile,
               xmin = -3, xmax = x_percent,
               color = colors[int(i)])
    
    plt.vlines(x = x_percent,
               ymin = 0, ymax = percentile,
               color = colors[int(i)])
    
    plt.plot(x_percent, percentile, marker = 'x',color = colors[int(i)],
             markersize = 12)

ax.set_xlim(-3, 3)
ax.set_ylim(0,  1)
ax.set_xlabel('z')
ax.set_ylabel('CDF, $F_{Z}(z)$')
