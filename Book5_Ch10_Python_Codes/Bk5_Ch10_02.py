

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
from scipy.stats import multivariate_normal
from scipy.stats import norm

rho     = 0.75 # -0.75, 0, 0.75
sigma_X = 1 # 1, 2
sigma_Y = 1 # 1, 2

mu_X = 0
mu_Y = 0
mu    = [mu_X, mu_Y]

Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], 
        [sigma_X*sigma_Y*rho, sigma_Y**2]]

width = 4
X = np.arange(-width,width,0.05)
Y = np.arange(-width,width,0.05)

XX, YY = np.meshgrid(X, Y)

XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(mu, Sigma)

# visualize PDF

f_X_Y_joint = bi_norm.pdf(XXYY)

# Plot the conditional distributions
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, 
                       width_ratios=[3, 1], 
                       height_ratios=[3, 1])

# # gs.update(wspace=0., hspace=0.)
# plt.suptitle('Marginal distributions', y=0.93)

# Plot surface on top left
ax1 = plt.subplot(gs[0])

# Plot bivariate normal
ax1.contourf(XX, YY, f_X_Y_joint, 15, cmap=cm.RdYlBu_r)
ax1.axvline(x = mu_X, color = 'k', linestyle = '--')
ax1.axhline(y = mu_Y, color = 'k', linestyle = '--')

ax1.set_xlabel('$X$')
ax1.set_ylabel('$Y$')
ax1.yaxis.set_label_position('right')
ax1.set_xticks([])
ax1.set_yticks([])

# Plot Y marginal
ax2 = plt.subplot(gs[1])
f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)

ax2.plot(f_Y, Y, 'b', label='$f_{Y}(y)$')
ax2.axhline(y = mu_Y, color = 'r', linestyle = '--')

ax2.fill_between(f_Y,Y, 
                 edgecolor = 'none', 
                 facecolor = '#DBEEF3')
ax2.legend(loc=0)
ax2.set_xlabel('PDF')
ax2.set_ylim(-width, width)
ax2.set_xlim(0, 0.5)
ax2.invert_xaxis()
ax2.yaxis.tick_right()

# Plot X marginal
ax3 = plt.subplot(gs[2])
f_X = norm.pdf(X, loc=mu_X, scale=sigma_X)

ax3.plot(X, f_X, 'b', label='$f_{X}(x)$')
ax3.axvline(x = mu_X, color = 'r', linestyle = '--')

ax3.fill_between(X,f_X, 
                 edgecolor = 'none', 
                 facecolor = '#DBEEF3')
ax3.legend(loc=0)
ax3.set_ylabel('PDF')
ax3.yaxis.set_label_position('left')
ax3.set_xlim(-width, width)
ax3.set_ylim(0, 0.5)

ax4 = plt.subplot(gs[3])
ax4.set_visible(False)

plt.show()
