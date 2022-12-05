

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
from mpl_toolkits.mplot3d import axes3d

def fcn_Y_given_X (mu_X, mu_Y, sigma_X, sigma_Y, rho, X, Y):
    
    coeff = 1/sigma_Y/np.sqrt(1 - rho**2)/np.sqrt(2*np.pi)
    sym_axis = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)
    
    quad  = -1/2*((Y - sym_axis)/sigma_Y/np.sqrt(1 - rho**2))**2
    
    f_Y_given_X  = coeff*np.exp(quad)
    
    return f_Y_given_X

# parameters

rho     = 0.5
sigma_X = 1
sigma_Y = 1

mu_X = 0
mu_Y = 0

width = 3
X = np.linspace(-width,width,31)
Y = np.linspace(-width,width,31)

XX, YY = np.meshgrid(X, Y)

f_Y_given_X = fcn_Y_given_X (mu_X, mu_Y, sigma_X, sigma_Y, rho, XX, YY)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(XX, YY, f_Y_given_X,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{Y|X}(y|x)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()

#%% surface projected along X to Y-Z plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, YY, f_Y_given_X, rstride=0, cstride=1,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour(XX, YY, f_Y_given_X, 
           levels = 20, zdir='x', \
            offset=YY.max(), cmap=cm.RdYlBu_r)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{Y|X}(y|x)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()


# add X marginal

f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)

fig, ax = plt.subplots()

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(X)))

for i in np.linspace(1,len(X),len(X)):
    plt.plot(Y,f_Y_given_X[:,int(i)-1],
             color = colors[int(i)-1])

plt.plot(Y,f_Y, color = 'k')

plt.xlabel('y')
plt.ylabel('$f_{Y|X}(y|x)$')
ax.set_xlim(-width, width)
ax.set_ylim(0, f_Y_given_X.max())

#%% surface projected along Z to X-Y plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, YY, f_Y_given_X,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour3D(XX,YY,f_Y_given_X,12,
              cmap = 'RdYlBu_r')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{Y|X}(y|x)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()

# Plot filled contours

E_Y_given_X = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)

from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(7, 7))

# Plot bivariate normal
plt.contourf(XX, YY, f_Y_given_X, 12, cmap=cm.RdYlBu_r)
plt.plot(X,E_Y_given_X, color = 'k', linewidth = 1.25)
plt.axvline(x = mu_X, color = 'k', linestyle = '--')
plt.axhline(y = mu_Y, color = 'k', linestyle = '--')

x = np.linspace(-width,width,num = 201)
y = np.linspace(-width,width,num = 201)

xx,yy = np.meshgrid(x,y);

ellipse = ((xx/sigma_X)**2 - 
           2*rho*(xx/sigma_X)*(yy/sigma_Y) + 
           (yy/sigma_Y)**2)/(1 - rho**2);

plt.contour(xx,yy,ellipse,levels = [1], colors = 'k')

rect = Rectangle(xy = [- sigma_X, - sigma_Y] , 
                 width = 2*sigma_X, 
                 height = 2*sigma_Y,
                 edgecolor = 'k',facecolor="none")

ax.add_patch(rect)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
