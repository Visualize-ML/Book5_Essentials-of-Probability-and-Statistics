

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats.distributions import chi2

intervals = np.linspace(0.9, 0.99, 11);

np.sqrt(chi2.ppf((0.9,0.95,0.99), df=1))
dist_chi2_sqrt = np.sqrt(chi2.ppf(intervals, df=2));

num_sigma = np.linspace(1,3,3)

prob = chi2.cdf(num_sigma**2, df=2)

x = np.linspace(0,4,100) # mahal d

fig, ax = plt.subplots(figsize=(8, 8))

for df in [1,2,3,4,5,6]:
    
    prob_x_df_D = chi2.cdf(x**2, df=df)
    
    plt.plot(x,prob_x_df_D, label = 'df = ' + str(df))

plt.grid(color = (0.8,0.8,0.8))
plt.legend()
plt.yticks(np.linspace(0,1,21))
plt.xticks(np.linspace(0,4,21))

plt.xlim(0,4)
plt.ylim(0,1)
plt.xlabel('Mahal d ($\sigma$)')
plt.ylabel('$\u03B1$')
