

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import logistic
from matplotlib import cm # Colormaps

x = np.linspace(start = -5, stop = 5, num = 200)

# plot PDF curves

fig, ax = plt.subplots()

Ss = np.arange(0.5,2.1,0.1)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(Ss)))

for i in range(0,len(Ss)):
    s = Ss[i]
    plt.plot(x, logistic.pdf(x, loc = 0, scale = s), 
             color = colors[int(i)],
             label = "s = %.1f" %s)

ax.axvline(x = 0, color = 'k', linestyle = '--')

plt.ylim((0, 0.5))
plt.xlim((-5,5))
plt.title("PDF of logistic distribution")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves
fig, ax = plt.subplots()

for i in range(0,len(Ss)):
    s = Ss[i]
    plt.plot(x, logistic.cdf(x, loc = 0, scale = s), 
             color = colors[int(i)],
             label = "s = %.1f" %s)

ax.axvline(x = 0, color = 'k', linestyle = '--')
plt.ylim((0, 1))
plt.xlim((-5,5))
plt.title("CDF of logistic distribution")
plt.ylabel("CDF")
plt.legend()
plt.show()

