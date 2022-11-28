

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import lognorm
from matplotlib import cm # Colormaps

x = np.linspace(start = 0, stop = 10, num = 500)

# plot PDF curves

fig, ax = plt.subplots()

STDs = np.arange(0.5,2.1,0.1)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(STDs)))

for i in range(0,len(STDs)):
    std = STDs[i]
    plt.plot(x, lognorm.pdf(x, loc = 0, s = std), 
             color = colors[int(i)],
             label = "\u03C3= %.1f" %std)

plt.ylim((0, 1.5))
plt.xlim((0,10))
plt.title("PDF of lognormal")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves
fig, ax = plt.subplots()

for i in range(0,len(STDs)):
    std = STDs[i]
    plt.plot(x, lognorm.cdf(x, loc = 0, s = std), 
             color = colors[int(i)],
             label = "\u03C3= %.1f" %std)

plt.ylim((0, 1))
plt.xlim((0,10))
plt.title("CDF of lognormal")
plt.ylabel("CDF")
plt.legend()
plt.show()
