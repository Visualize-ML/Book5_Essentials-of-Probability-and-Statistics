

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import load_iris
from scipy.stats import norm
import scipy
import seaborn as sns
from numpy.linalg import inv

iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

x_array = np.linspace(0,8,100)
# Convert X array to dataframe
X_Y_df = pd.DataFrame(X, columns=feature_names)

#%% Heatmap of centroid vector, MU

MU = X_Y_df.mean()
MU = np.array([MU]).T

fig, axs = plt.subplots()

h = sns.heatmap(MU,cmap='RdYlBu_r', 
                linewidths=.05,annot=True,fmt = '.2f')

h.set_aspect("equal")
h.set_title('Vector of $\mu$')

#%% Heatmap of covariance matrix

SIGMA = X_Y_df.cov()

fig, axs = plt.subplots()

h = sns.heatmap(SIGMA,cmap='RdYlBu_r', linewidths=.05,annot=True)
h.set_aspect("equal")
h.set_title('Covariance matrix')

#%% 

SIGMA = np.array(SIGMA)

from sympy import symbols
x1, x2, x3 = symbols('x1 x2 x3')

SIGMA_XX = SIGMA[0:3,0:3]

SIGMA_YX = SIGMA[3,0:3]
SIGMA_YX = np.matrix(SIGMA_YX)

MU_Y = MU[3]
MU_Y = np.matrix(MU_Y)

MU_X = MU[0:3]

x_vec = np.array([[x1,x2,x3]]).T

y = SIGMA_YX@inv(SIGMA_XX)@(x_vec - MU_X) + MU_Y

print(y)

#%% matrix computation

b  = SIGMA_YX@inv(SIGMA_XX) # coefficients
b0 = MU_Y - b@MU_X          # constant

# computate coefficient vector

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(b, cmap ='RdYlBu_r', 
                 linewidths=.05,annot=True,
                 cbar_kws={"orientation": "horizontal"},fmt = '.3f',
                 vmax = 10, vmin = -5)

ax.set_aspect("equal")
plt.title('$b$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(SIGMA_YX,cmap='RdYlBu_r', linewidths=.05,annot=True,
                 cbar_kws={"orientation": "horizontal"},fmt = '.3f',
                 vmax = 10, vmin = -5)
ax.set_aspect("equal")
plt.title('$\Sigma_{YX}$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(inv(SIGMA_XX),cmap='RdYlBu_r', linewidths=.05,annot=True,
                 cbar_kws={"orientation": "horizontal"},fmt = '.3f',
                 vmax = 10, vmin = -5)
ax.set_aspect("equal")
plt.title('$\Sigma^{-1}_{XX}$')

# compute constant b0

fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(b0, cmap ='RdYlBu_r', 
                 linewidths=.05,annot=True,
                 cbar_kws={"orientation": "horizontal"},fmt = '.3f',
                 vmax = 5, vmin = -0.5)

ax.set_aspect("equal")
plt.title('$b_0$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(MU_Y,cmap='RdYlBu_r', linewidths=.05,annot=True,
                 cbar_kws={"orientation": "horizontal"},fmt = '.3f',
                 vmax = 5, vmin = -0.5)
ax.set_aspect("equal")
plt.title('$\mu_Y$')

plt.sca(axs[3])
plt.title('-')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(b,cmap='RdYlBu_r', linewidths=.05,annot=True,
                 cbar_kws={"orientation": "horizontal"},fmt = '.3f',
                 vmax = 5, vmin = -0.5)
ax.set_aspect("equal")
plt.title('$b$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(MU_X,cmap='RdYlBu_r', linewidths=.05,annot=True,
                 cbar_kws={"orientation": "horizontal"},fmt = '.3f',
                 vmax = 5, vmin = -0.5)
ax.set_aspect("equal")
plt.title('$\mu_X$')

#%% use statsmodels

import statsmodels.api as sm

X_df = X_Y_df[feature_names[0:3]]
y_df = X_Y_df[feature_names[3]]

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params
print(p)
