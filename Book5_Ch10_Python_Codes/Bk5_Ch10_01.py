

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm # Colormaps

rho     = 0.75
sigma_X = 1
sigma_Y = 2
mu_X = 0
mu_Y = 0

mu    = [mu_X, mu_Y]
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], 
        [sigma_X*sigma_Y*rho, sigma_Y**2]]

width = 4
X = np.linspace(-width,width,321)
Y = np.linspace(-width,width,321)

XX, YY = np.meshgrid(X, Y)

XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(mu, Sigma)

#%% visualize joint PDF surface

f_X_Y_joint = bi_norm.pdf(XXYY)

# 3D visualization

fig, ax = plt.subplots(1,2)
ax = plt.axes(projection='3d')

ax.plot_wireframe(XX,YY, f_X_Y_joint,
                  rstride=10, cstride=10,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{X,Y}(x,y)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
plt.show()


#%% surface projected along Y to X-Z plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, YY, f_X_Y_joint, rstride=10, cstride=0,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour(XX, YY, f_X_Y_joint, 
           levels = 33, zdir='y', \
            offset=XX.max(), cmap=cm.RdYlBu_r)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{X,Y}(x,y)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(X)))

for i in np.arange(1,len(X),5):
    plt.plot(X,f_X_Y_joint[int(i)-1,:],
             color = colors[int(i)-1])

plt.xlabel('x')
plt.ylabel('$f_{X,Y}(x,y)$')
ax.set_xlim(-width, width)
ax.set_ylim(0, f_X_Y_joint.max())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%% surface projected along Y to Y-Z plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, YY, f_X_Y_joint, rstride=0, cstride=10,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour(XX, YY, f_X_Y_joint, 
           levels = 33, zdir='x', \
            offset=YY.max(), cmap=cm.RdYlBu_r)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{X,Y}(x,y)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(Y)))

for i in np.arange(1,len(X),5):
    plt.plot(X,f_X_Y_joint[:,int(i)-1],
             color = colors[int(i)-1])

plt.xlabel('y')
plt.ylabel('$f_{X,Y}(x,y)$')
ax.set_xlim(-width, width)
ax.set_ylim(0, f_X_Y_joint.max())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%% surface projected along Z to X-Y plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, YY, f_X_Y_joint,
                  rstride=10, cstride=10,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour3D(XX,YY, f_X_Y_joint,15,
             cmap = 'RdYlBu_r')

# ax.contourf(XX, YY, f_X_Y_joint, levels = 12, zdir='z', \
#             offset=0, cmap=cm.RdYlBu_r)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{X,Y}(x,y)$')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
plt.tight_layout()
plt.show()

# Plot filled contours

fig, ax = plt.subplots(figsize=(7, 7))

# Plot bivariate normal
plt.contourf(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
plt.axvline(x = mu_X, color = 'r', linestyle = '--')
plt.axhline(y = mu_Y, color = 'r', linestyle = '--')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
