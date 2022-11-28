

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.stats import geom
import matplotlib.pyplot as plt
import numpy as np

p = 0.5

mean,var,skew,kurt = geom.stats(p,moments='mvsk')
print('Expectation, Variance, Skewness, Kurtosis: ', mean, var, skew, kurt)

k_range = np.arange(1,15 + 1)
# PMF versus x

fig, ax = plt.subplots()

plt.stem(k_range, geom.pmf(k_range, p))

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(1,k_range.max())
plt.ylim(0,p)

# CDF versus x

fig, ax = plt.subplots()

plt.stem(k_range, geom.cdf(k_range, p))
plt.axhline(y = 1, color = 'r', linestyle = '--')

plt.xlabel('x')
plt.ylabel('CDF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(1,k_range.max())
plt.ylim(0,1)

# PMF versus x as p varies

k_range = np.arange(1,16)

fig, ax = plt.subplots()

for p in [0.4,0.5,0.6,0.7,0.8]:
    plt.plot(k_range, geom.pmf(k_range, p), 
             marker = 'x',markersize = 12,
             label = 'p = ' + str(p))

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(1,k_range.max())
plt.ylim(0,0.8)
plt.xticks([1,5,10,15])
plt.legend()
