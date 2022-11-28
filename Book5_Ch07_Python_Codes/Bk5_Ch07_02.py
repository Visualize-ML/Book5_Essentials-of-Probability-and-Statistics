

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


x = np.linspace(-3, 5, num=100)

# plot PDF
fig = plt.figure()

mean     = 0
variance = 1

plt.plot(x, norm.pdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(0, 1)$")

# The location (loc) keyword specifies the mean. 
# The scale (scale) keyword specifies the standard deviation.
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = 2
variance = 3

plt.plot(x, norm.pdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(2, 3)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = -1
variance = 0.5

plt.plot(x, norm.pdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(2, 3)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

plt.xlabel('$x$')
plt.ylabel('PDF, $f(x)$')

plt.ylim([0, 1])
plt.xlim([-3, 5])
plt.legend(loc=1)

# plot CDF curves
fig = plt.figure()

mean     = 0
variance = 1

plt.plot(x, norm.cdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(0, 1)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = 2
variance = 3

plt.plot(x, norm.cdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(2, 3)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = -1
variance = 0.5

plt.plot(x, norm.cdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(-1, 0.5)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

plt.axhline(y = 0.5, linestyle = '--', color = 'r')

plt.xlabel('$x$')
plt.ylabel('CDF, $F(x)$')

plt.ylim([0, 1])
plt.xlim([-3, 5])
plt.legend(loc=4)
