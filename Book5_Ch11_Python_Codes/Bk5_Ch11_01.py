
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm # Colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import Rectangle

# parameters

rho     = 0.5
sigma_X = 1
sigma_Y = 1.5

mu_X = 0
mu_Y = 0

width = 3

mu    = [mu_X, mu_Y]
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], 
        [sigma_X*sigma_Y*rho, sigma_Y**2]]

X = np.linspace(-width,width,101)
Y = np.linspace(-width,width,101)

XX, YY = np.meshgrid(X, Y)

XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(mu, Sigma)

f_X_Y_joint = bi_norm.pdf(XXYY)

# expectation of Y given X

E_Y_given_X = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)

# expectation of X given Y

E_X_given_Y = mu_X + rho*sigma_X/sigma_Y*(Y - mu_Y)

theta = 1/2*np.arctan(2*rho*sigma_X*sigma_Y/(sigma_X**2 - sigma_Y**2))
k = np.tan(theta)

axis_minor = mu_Y + k*(X - mu_X)
axis_major = mu_Y - 1/k*(X - mu_X)

fig, ax = plt.subplots(figsize=(7, 7))

# Plot bivariate normal
plt.contour(XX, YY, f_X_Y_joint, 25, cmap=cm.RdYlBu_r)
plt.axvline(x = mu_X, color = 'k', linestyle = '--')
plt.axhline(y = mu_Y, color = 'k', linestyle = '--')

plt.plot(E_X_given_Y,Y, color = 'k', linewidth = 1.25)
plt.plot(X,E_Y_given_X, color = 'k', linewidth = 1.25)

# plot ellipse minor and major axes
plt.plot(X,axis_minor, color = 'r', linewidth = 1.25)
plt.plot(X,axis_major, color = 'r', linewidth = 1.25)

rect = Rectangle(xy = [mu_X - sigma_X, mu_Y - sigma_Y] , 
                 width = 2*sigma_X, 
                 height = 2*sigma_Y,
                 edgecolor = 'k',facecolor="none")

ax.add_patch(rect)

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_xlim(-width,width)
ax.set_ylim(-width,width)
