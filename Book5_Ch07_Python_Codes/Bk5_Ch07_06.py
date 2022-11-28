

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
from scipy.stats import norm, lognorm

def logn_pdf(x,mu,sigma):
    
    scaling = 1/x/sigma/np.sqrt(2*np.pi)
    
    exp_part = np.exp(-(np.log(x) - mu)**2/2/sigma**2)
    
    pdf = scaling*exp_part
    
    return pdf


width = 4
X = np.linspace(-3,3,200)
Y = np.linspace(0.01,10,200)

mu    = 0
sigma = 1
pdf_X = norm.pdf(X, mu, sigma)
pdf_Y = lognorm.pdf(Y,s = sigma, scale = np.exp(mu))

mu_Y, var_Y, skew_Y, kurt_Y = lognorm.stats(s = sigma, 
                                      scale = np.exp(mu), 
                                      moments='mvsk')
# mu_Y = np.exp(mu + sigma**2/2)

pdf_Y_2 = logn_pdf(Y,mu,sigma)

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
ax1.plot(X, np.exp(X))
ax1.axvline(x = 0, color = 'k', linestyle = '--')
ax1.axhline(y = 1, color = 'k', linestyle = '--')

ax1.axvline(x = mu, color = 'r', linestyle = '--')
ax1.axhline(y = mu_Y, color = 'k', linestyle = '--')
ax1.axhline(y = np.exp(mu), color = 'k', linestyle = '--')

ax1.set_xlabel('$X$')
ax1.set_ylabel('$Y$')
ax1.yaxis.set_label_position('right')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim(X.min(),X.max())
ax1.set_ylim(Y.min(),Y.max())

# Plot Y marginal
ax2 = plt.subplot(gs[1])

ax2.plot(pdf_Y, Y, 'b', label='$f_{Y}(y)$')

ax2.plot(pdf_Y_2, Y, 'k')
ax2.axhline(y = mu_Y, color = 'r', linestyle = '--')
ax2.axhline(y = np.exp(mu), color = 'r', linestyle = '--')

ax2.fill_between(pdf_Y,Y, 
                 edgecolor = 'none', 
                 facecolor = '#DBEEF3')
ax2.legend(loc=0)
ax2.set_xlabel('PDF')
ax2.set_ylim(Y.min(),Y.max())
ax2.set_xlim(0, pdf_Y.max()*1.1)
ax2.invert_xaxis()
ax2.yaxis.tick_right()

# Plot X marginal
ax3 = plt.subplot(gs[2])

ax3.plot(X, pdf_X, 'b', label='$f_{X}(x)$')
ax3.axvline(x = mu, color = 'r', linestyle = '--')

ax3.fill_between(X,pdf_X, 
                 edgecolor = 'none', 
                 facecolor = '#DBEEF3')
ax3.legend(loc=0)
ax3.set_ylabel('PDF')
ax3.yaxis.set_label_position('left')
ax3.set_xlim(X.min(),X.max())
ax3.set_ylim(0, pdf_X.max()*1.1)


ax4 = plt.subplot(gs[3])
ax4.set_visible(False)

plt.show()

