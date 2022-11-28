

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import expon
from matplotlib import cm # Colormaps

x = np.linspace(start = 0, stop = 10, num = 500)

# plot PDF curves

fig, ax = plt.subplots()

lambdas = np.arange(0.1,1.1,0.1)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(lambdas)))

for i in range(0,len(lambdas)):
    lambda_i = lambdas[i]
    plt.plot(x, expon.pdf(x, loc = 0, scale = 1/lambda_i), 
             color = colors[int(i)],
             label = "\u03BB= %.2f" %lambda_i)

plt.ylim((0, 1))
plt.xlim((0,10))
plt.title("PDF of exponential distribution")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves
fig, ax = plt.subplots()

for i in range(0,len(lambdas)):
    lambda_i = lambdas[i]
    plt.plot(x, expon.cdf(x, loc = 0, scale = 1/lambda_i), 
             color = colors[int(i)],
             label = "\u03BB= %.2f" %lambda_i)

plt.ylim((0, 1))
plt.xlim((0,10))
plt.title("CDF of exponential distribution")
plt.ylabel("CDF")
plt.legend()
plt.show()
