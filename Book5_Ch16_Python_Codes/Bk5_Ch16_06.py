

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x   = np.linspace(start = 0, stop = 16, num = 1000)
# In practice, population standard deviation is rarely known
n = 6
f_x_chi2 = stats.chi2.pdf(x, df = n-1) 


alpha = 0.05

#%% Get the critical values, two-tailed

crit_value_right = stats.chi2.ppf(q = 1-alpha/2, df = n-1)
crit_value_left  = stats.chi2.ppf(q = alpha/2, df = n-1)

fig, ax = plt.subplots()

plt.plot(x, f_x_chi2, color = "#0070C0")

plt.fill_between(x[np.logical_and(x >= crit_value_left, x <= crit_value_right)], 
                 f_x_chi2[np.logical_and(x >= crit_value_left, x <= crit_value_right)], 
                 color = "#DBEEF3")

ax.axvline(x = crit_value_right,  color = 'r', linestyle = '--')
plt.plot(crit_value_right, 0,marker = 'x', color = 'k', markersize = 12)
ax.axvline(x = crit_value_left, color = 'r', linestyle = '--')
plt.plot(crit_value_left,0,marker = 'x', color = 'k', markersize = 12)

plt.fill_between(x[np.logical_and(x >= 0, x <= crit_value_left)], 
                 f_x_chi2[np.logical_and(x >= 0, x <= crit_value_left)], 
                 color = "#FF9980")

plt.fill_between(x[np.logical_and(x <= x.max(), x >= crit_value_right)], 
                 f_x_chi2[np.logical_and(x <= x.max(), x >= crit_value_right)], 
                 color = "#FF9980")

ax.set_xlim(0,16)
ax.set_ylim(0,0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
