

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x   = np.linspace(start = -4, stop = 4, num = 200)
# In practice, population standard deviation is rarely known
n = 6
f_x_t = stats.t.pdf(x, df = n-1) 
# PDF of student t distribution
f_x_norm = stats.norm.pdf(x) 

alpha = 0.05

#%% Get the critical values, two-tailed

crit_value = stats.t.ppf(q = 1-alpha/2, df = n-1)

fig, ax = plt.subplots()

plt.plot(x, f_x_t, color = "#0070C0")
plt.plot(x, f_x_norm, color = "k", linestyle = '--')

plt.fill_between(x[np.logical_and(x >= -crit_value, x <= crit_value)], 
                 f_x_t[np.logical_and(x >= -crit_value, x <= crit_value)], 
                 color = "#DBEEF3")

ax.axvline(x = crit_value,  color = 'r', linestyle = '--')
plt.plot(crit_value, 0,marker = 'x', color = 'k', markersize = 12)
ax.axvline(x = -crit_value, color = 'r', linestyle = '--')
plt.plot(-crit_value,0,marker = 'x', color = 'k', markersize = 12)

plt.fill_between(x[x <= -crit_value], f_x_t[x <= -crit_value], color = "#FF9980")
plt.fill_between(x[x >= crit_value],  f_x_t[x >= crit_value], color = "#FF9980")

plt.title("Population sigma unknown, $\\alpha = 0.05$, two-tailed")

ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
