

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
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

rho     = 0.5
sigma_X = 1.5
sigma_Y = 1

mu_X = 0
mu_Y = 0
mu    = [mu_X, mu_Y]

Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], 
        [sigma_X*sigma_Y*rho, sigma_Y**2]]

width = 4
X = np.linspace(-width,width,81)
Y = np.linspace(-width,width,81)

XX, YY = np.meshgrid(X, Y)

XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(mu, Sigma)

#%% visualize PDF

y_cond_i = 60 # 20, 30, 40, 50, 60, index


f_X_Y_joint = bi_norm.pdf(XXYY)

# Plot the tional distributions
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, 
                       width_ratios=[3, 1], 
                       height_ratios=[3, 1])

# Plot surface on top left
ax1 = plt.subplot(gs[0])

# Plot bivariate normal
ax1.contour(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
ax1.axvline(x = mu_X, color = 'k', linestyle = '--')
ax1.axhline(y = mu_Y, color = 'k', linestyle = '--')
ax1.axhline(y = Y[y_cond_i], color = 'r', linestyle = '--')

x_sym_axis = mu_X + rho*sigma_X/sigma_Y*(Y[y_cond_i] - mu_Y)
ax1.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

ax1.set_xlabel('$X$')
ax1.set_ylabel('$Y$')
ax1.yaxis.set_label_position('right')
ax1.set_xticks([])
ax1.set_yticks([])

# Plot Y marginal
ax2 = plt.subplot(gs[1])
f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)

ax2.plot(f_Y, Y, 'k', label='$f_{Y}(y)$')
ax2.axhline(y = mu_Y, color = 'k', linestyle = '--')
ax2.axhline(y = Y[y_cond_i], color = 'r', linestyle = '--')
ax2.plot(f_Y[y_cond_i], Y[y_cond_i], marker = 'x', markersize = 15)
plt.title('$f_{Y}(y_{} = %.2f) = %.2f$'
          %(Y[y_cond_i],f_Y[y_cond_i]))

ax2.fill_between(f_Y,Y, 
                 edgecolor = 'none', 
                 facecolor = '#D9D9D9')
ax2.legend(loc=0)
ax2.set_xlabel('PDF')
ax2.set_ylim(-width, width)
ax2.set_xlim(0, 0.5)
ax2.invert_xaxis()
ax2.yaxis.tick_right()

# Plot X and Y joint

ax3 = plt.subplot(gs[2])
f_X_Y_cond_i = f_X_Y_joint[y_cond_i,:]

ax3.plot(X, f_X_Y_cond_i, 'r', 
         label='$f_{X,Y}(x,y_{} = %.2f)$' %(Y[y_cond_i]))


ax3.axvline(x = mu_X, color = 'k', linestyle = '--')
ax3.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

ax3.legend(loc=0)
ax3.set_ylabel('PDF')
ax3.yaxis.set_label_position('left')
ax3.set_xlim(-width, width)
ax3.set_ylim(0, 0.5)
ax3.set_yticks([0, 0.25, 0.5])

ax4 = plt.subplot(gs[3])
ax4.set_visible(False)

plt.show()

#%% compare joint, marginal and tional

f_X = norm.pdf(X, loc=mu_X, scale=sigma_X)

fig, ax = plt.subplots()

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(Y)))

f_X_given_Y_cond_i = f_X_Y_cond_i/f_Y[y_cond_i]

plt.plot(X,f_X, color = 'k',
         label='$f_{X}(x)$') # marginal
ax.axvline(x = mu_X, color = 'k', linestyle = '--')


plt.plot(X,f_X_Y_cond_i, color = 'r',
         label='$f_{X,Y}(x,y_{} = %.2f$)' %(Y[y_cond_i])) # joint
ax.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

plt.plot(X,f_X_given_Y_cond_i, color = 'b',
         label='$f_{X|Y}(x|y_{} = %.2f$)' %(Y[y_cond_i])) # tional

ax.fill_between(X,f_X_given_Y_cond_i, 
                edgecolor = 'none', 
                facecolor = '#DBEEF3')

ax.fill_between(X,f_X_Y_cond_i, 
                edgecolor = 'none',
                hatch='/')


plt.xlabel('X')
plt.ylabel('PDF')
ax.set_xlim(-width, width)
ax.set_ylim(0, 0.35)
plt.title('$f_{Y}(y_{} = %.2f) = %.2f$'
          %(Y[y_cond_i],f_Y[y_cond_i]))
ax.legend()
