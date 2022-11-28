

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import chi2
from matplotlib import cm # Colormaps

x = np.linspace(start = 0, stop = 10, num = 200)

# plot PDF curves

fig, ax = plt.subplots()

DFs = range(1,10)
colors = plt.cm.RdYlBu(np.linspace(0,1,len(DFs)))

for df in DFs:

    plt.plot(x, chi2.pdf(x, df = df), 
             color = colors[int(df)-1],
             label = "k = " + str(df))

plt.ylim((0, 1))
plt.xlim((0, 10))
plt.title("PDF of $\\chi^2_k$")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves

fig, ax = plt.subplots()

DFs = range(1,10)
colors = plt.cm.RdYlBu(np.linspace(0,1,len(DFs)))

for df in DFs:

    plt.plot(x, chi2.cdf(x, df = df), 
             color = colors[int(df)-1],
             label = "k = " + str(df))

plt.ylim((0, 1))
plt.xlim((0, 10))
plt.title("CDF of $\\chi^2_k$")
plt.ylabel("CDF")
plt.legend()
plt.show()

