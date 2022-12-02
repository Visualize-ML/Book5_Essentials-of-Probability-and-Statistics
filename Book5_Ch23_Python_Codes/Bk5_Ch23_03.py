

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############



import numpy as np
from sklearn.covariance import EmpiricalCovariance

SIGMA = np.array([[5, 3],[3, 5]])/8;

mu = np.array([0, 0]);

R1, R2 = np.random.multivariate_normal(mu, SIGMA, 1000).T

x1 = np.linspace(-3,3,100);
x2 = x1;

[X1,X2] = np.meshgrid(x1,x2);

X = np.array([X1.flatten(), X2.flatten()]).T;

#%% Mahal distance mesh

emp_cov_Xc = EmpiricalCovariance().fit(np.vstack((R1,R2)).T)

mahal_sq_Xc = emp_cov_Xc.mahalanobis(X)

mahal_sq_Xc = mahal_sq_Xc.reshape(X1.shape)
mahal_d_Xc = np.sqrt(mahal_sq_Xc)

#%%

import matplotlib.pyplot as plt

levels = np.linspace(1,5,9);

fig, ax = plt.subplots()

ax.contour(X1, X2, mahal_d_Xc, levels = levels, cmap = 'rainbow')
plt.scatter(R1,R2,s = 6, color = [0.5, 0.5, 0.5])

ax.axhline(y=0, color='k', linewidth = 0.25)
ax.axvline(x=0, color='k', linewidth = 0.25)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.axis('scaled')

#%%

intervals = np.linspace(0.9,0.99,10);

from scipy.stats.distributions import chi2

dist_chi2_sqrt = np.sqrt(chi2.ppf(intervals, df=2));

fig, ax = plt.subplots()

ax.contour(X1, X2, mahal_d_Xc, levels = dist_chi2_sqrt, cmap = 'rainbow')
plt.scatter(R1,R2,s = 6, color = [0.5, 0.5, 0.5])

ax.axhline(y=0, color='k', linewidth = 0.25)
ax.axvline(x=0, color='k', linewidth = 0.25)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.axis('scaled')
