

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
from scipy.stats import norm

# varying mu

X  = np.arange(-5,5,0.05)
mu = np.arange(-2,2,0.2)

XX, MM = np.meshgrid(X, mu)

sigma = 1

F_x_varying_mu = norm.cdf(XX, loc=MM, scale=sigma)

# surface projected along Y to X-Z plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, MM, F_x_varying_mu, rstride=1, cstride=0,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour(XX, MM, F_x_varying_mu, 
           levels = 20, zdir='y', \
            offset=mu.max(), cmap=cm.RdYlBu_r)

ax.set_xlabel('$X$')
ax.set_ylabel('$\mu$')
ax.set_zlabel('$F_{X}(x)$')

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)

ax.view_init(azim=-120, elev=30)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(mu)))

for i in np.linspace(1,len(mu),len(mu)):
    plt.plot(X,F_x_varying_mu[int(i)-1,:],
             color = colors[int(i)-1])

plt.axhline(y = 0.5, color = 'k', linestyle = '--')
plt.axhline(y = 1,   color = 'k', linestyle = '--')

plt.xlabel('X')
plt.ylabel('$F_{X}(x)$')
ax.set_xlim(-5, 5)
ax.set_ylim(0, F_x_varying_mu.max())

#%% Varying sigma

X     = np.arange(-5,5,0.05)
sigma = np.arange(0.5,3,0.1)

XX, SS = np.meshgrid(X, sigma)

mu = 0

F_x_varying_sig = norm.cdf(XX, loc=mu, scale=SS)

# F_x_varying_sig = np.cumsum(f_x_varying_sig,axis = 1)*0.05

# surface projected along Y to X-Z plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, SS, F_x_varying_sig, rstride=1, cstride=0,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour(XX, SS, F_x_varying_sig, 
           levels = 20, zdir='y', \
            offset=sigma.max(), cmap=cm.RdYlBu_r)

ax.set_xlabel('$X$')
ax.set_ylabel('$\sigma$')
ax.set_zlabel('$F_{X}(x)$')

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-5, 5)
ax.set_ylim(0.5,3)

ax.view_init(azim=-120, elev=30)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(sigma)))

for i in np.linspace(1,len(sigma),len(sigma)):
    plt.plot(X,F_x_varying_sig[int(i)-1,:],
             color = colors[int(i)-1])

plt.axhline(y = 0.5, color = 'k', linestyle = '--')
plt.axhline(y = 1,   color = 'k', linestyle = '--')

plt.xlabel('X')
plt.ylabel('$F_{X}(x)$')
ax.set_xlim(-5, 5)
ax.set_ylim(0, F_x_varying_mu.max())
