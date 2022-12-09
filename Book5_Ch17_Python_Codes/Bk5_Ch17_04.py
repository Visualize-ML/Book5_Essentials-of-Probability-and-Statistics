

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
f_x = stats.norm.pdf(x) # PDF of standard normal distribution

alpha = 0.05

# population standard deviation is known, or large sample size

#%% Get the critical values, two-tailed

crit_value = stats.norm.ppf(q = 1-alpha/2)

fig, ax = plt.subplots()

plt.plot(x, f_x, color = "#0070C0")

plt.fill_between(x[np.logical_and(x >= -crit_value, x <= crit_value)], 
                 f_x[np.logical_and(x >= -crit_value, x <= crit_value)], 
                 color = "#DBEEF3")

ax.axvline(x = crit_value,  color = 'r', linestyle = '--')
plt.plot(crit_value, 0,marker = 'x', color = 'k', markersize = 12)
ax.axvline(x = -crit_value, color = 'r', linestyle = '--')
plt.plot(-crit_value,0,marker = 'x', color = 'k', markersize = 12)

plt.fill_between(x[x <= -crit_value], f_x[x <= -crit_value], color = "#FF9980")
plt.fill_between(x[x >= crit_value],  f_x[x >= crit_value], color = "#FF9980")

plt.title("Population sigma known, $\\alpha = 0.05$, two-tailed")

ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%% left-sided

crit_value = stats.norm.ppf(q = 1 - alpha)

fig, ax = plt.subplots()

plt.plot(x, f_x, color = "#0070C0")

plt.fill_between(x[x >= -crit_value], 
                 f_x[x >= -crit_value], 
                 color = "#DBEEF3")

ax.axvline(x = -crit_value, color = 'r', linestyle = '--')
plt.plot(-crit_value,0,marker = 'x', color = 'k', markersize = 12)

plt.fill_between(x[x <= -crit_value], 
                 f_x[x <= -crit_value], 
                 color = "#FF9980")

plt.title("Population sigma known, $\\alpha = 0.05$, left-tailed")

ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%% right-sided

crit_value = stats.norm.ppf(q = 1 - alpha)

fig, ax = plt.subplots()

plt.plot(x, f_x, color = "#0070C0")

plt.fill_between(x[x <= crit_value], 
                 f_x[x <= crit_value], 
                 color = "#DBEEF3")

ax.axvline(x = crit_value, color = 'r', linestyle = '--')
plt.plot(crit_value,0,marker = 'x', color = 'k', markersize = 12)

plt.fill_between(x[x >= crit_value], 
                 f_x[x >= crit_value], 
                 color = "#FF9980")

plt.title("Population sigma known, $\\alpha = 0.05$, right-tailed")

ax.set_xlim(-4,4)
ax.set_ylim(0,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
