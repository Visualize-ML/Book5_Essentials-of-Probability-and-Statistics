

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import load_iris

plt.close('all')

iris = load_iris()
# A copy from Sklearn

X = iris.data
x = X[:, 0]
y = X[:, 1]


xmin, xmax = 4, 8
ymin, ymax = 1, 5

# Perform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
PDF_xy = np.reshape(kernel(positions).T, xx.shape)


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, PDF_xy, 
                  rstride=4, cstride=4,
                  color = [0.5,0.5,0.5],
                  linewidth = 0.25)


colorbar = ax.contour(xx,yy, PDF_xy,20,
             cmap = 'RdYlBu_r')

fig.colorbar(colorbar, ax=ax)

ax.set_xlabel('Sepal length, $X_1$')
ax.set_ylabel('Sepal width, $X_2$')
ax.set_zlabel('$f_{X1,X2}(x_1,x_2)$')

ax.set_proj_type('ortho')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
ax.view_init(azim=-135, elev=30)
ax.grid(False)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
# ax.set_zlim(0, 0.7)
plt.tight_layout()

plt.show()


fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Contourf plot
cfset = ax.contourf(xx, yy, PDF_xy, cmap='Blues')
cset = ax.contour(xx, yy, PDF_xy, colors='k')
plt.scatter(x,y,marker = 'x')

# Label plot
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Sepal length, $X_1$')
ax.set_ylabel('Sepal width, $X_2$')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
