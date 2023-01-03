

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from statsmodels.nonparametric.kde import kernel_switch
from itertools import islice

list(kernel_switch.keys())

# Create a figure
fig = plt.figure(figsize=(12, 5))

# Enumerate every option for the kernel
for i, (ker_name, ker_class) in enumerate(islice(kernel_switch.items(),8)):
    
    # Initialize the kernel object
    kernel = ker_class()
    
    # Sample from the domain
    domain = kernel.domain or [-3, 3]
    x_vals = np.linspace(*domain, num=2**10)
    y_vals = kernel(x_vals)

    # Create a subplot, set the title
    ax = fig.add_subplot(2, 4, i + 1)
    ax.set_title('Kernel function "{}"'.format(ker_name))
    ax.plot(x_vals, y_vals, lw=3, label='{}'.format(ker_name))
    ax.scatter([0], [0], marker='x', color='red')
    plt.grid(True, zorder=-5)
    ax.set_xlim(domain)
    
plt.tight_layout()

data = [-3,-2,0,2,2.5,3,4]
kde = sm.nonparametric.KDEUnivariate(data)

# Create a figure
fig = plt.figure(figsize=(12, 5))

# Enumerate every option for the kernel
for i, kernel in enumerate(islice(kernel_switch.keys(),8)):
    
    # Create a subplot, set the title
    ax = fig.add_subplot(2, 4, i + 1)
    ax.set_title('Kernel function "{}"'.format(kernel))
    
    # Fit the model (estimate densities)
    kde.fit(kernel=kernel, fft=False, bw=1.5)
    
    ax.fill_between(kde.support, kde.density, facecolor = '#DBEEF4')
    # Create the plot
    ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)
    ax.scatter(data, np.zeros_like(data), marker='x', color='red')
    plt.grid()
    ax.set_xlim([-6, 6])
    ax.set_ylim([0, 0.3])
    
plt.tight_layout()
