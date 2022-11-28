

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import t
from scipy.stats import norm
from matplotlib import cm # Colormaps

x = np.linspace(start = -5, stop = 5, num = 200)

# plot PDF curves

fig, ax = plt.subplots()

DFs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 30]

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(DFs)))

for i in range(0,len(DFs)):
    df = DFs[i]
    plt.plot(x, t.pdf(x, df = df, loc = 0, scale = 1), 
             color = colors[int(i)],
             label = "\u03BD = " + str(df))

ax.axvline(x = 0, color = 'k', linestyle = '--')
# compare with normal
plt.plot(x,norm.pdf(x,loc = 0, scale = 1), color = 'k', label = 'Normal')
plt.ylim((0, 0.5))
plt.xlim((-5,5))
plt.title("PDF of student's t")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves
fig, ax = plt.subplots()

for i in range(0,len(DFs)):
    df = DFs[i]
    plt.plot(x, t.cdf(x, df = df, loc = 0, scale = 1), 
             color = colors[int(i)],
             label = "\u03BD = " + str(df))

ax.axvline(x = 0, color = 'k', linestyle = '--')
ax.axhline(y = 0.5, color = 'k', linestyle = '--')

# compare with normal
plt.plot(x,norm.cdf(x,loc = 0, scale = 1), color = 'k', label = 'Normal')
plt.ylim((0, 1))
plt.xlim((-5,5))
plt.title("CDF of student's t")
plt.ylabel("CDF")
plt.legend()
plt.show()
