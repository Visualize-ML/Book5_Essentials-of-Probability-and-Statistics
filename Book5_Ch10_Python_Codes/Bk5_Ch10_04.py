

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as multi_norm
import numpy as np
from matplotlib.patches import Rectangle

np.random.seed(2)

rho = -0.9

mu_X = 0
mu_Y = 0

MU = [mu_X, mu_Y]

sigma_X = 1
sigma_Y = 1

# covariance
SIGMA = [[sigma_X**2, sigma_X*sigma_Y*rho], 
         [sigma_X*sigma_Y*rho, sigma_Y**2]] 

num = 500

X, Y = multi_norm(MU, SIGMA, num).T

center_X = np.mean(X)
center_Y = np.mean(Y)

fig, ax = plt.subplots(figsize=(8, 8))

# plot center of data
plt.plot(X,Y,'.', color = '#00448A', 
         alpha = 0.25, markersize = 10)

ax.axvline(x = 0, color = 'k', linestyle = '--')
ax.axhline(y = 0, color = 'k', linestyle = '--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim((-3,3))
ax.set_ylim((-3,3))
plt.show()
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])


from scipy.stats import multivariate_normal

X_grid = np.linspace(-3,3,200)
Y_grid = np.linspace(-3,3,200)

XX, YY = np.meshgrid(X_grid, Y_grid)

XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(MU, SIGMA)

# visualize PDF

pdf_fine = bi_norm.pdf(XXYY)

# 3D visualization

fig, ax = plt.subplots(figsize=(8, 8))
ax = plt.axes(projection='3d')

ax.plot_wireframe(XX,YY, pdf_fine,
                  cstride = 10, rstride = 10,
                  color = [0.7,0.7,0.7],
                  linewidth = 0.25)

ax.contour3D(XX,YY,pdf_fine,15,
             cmap = 'RdYlBu_r')

ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PDF')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim3d([-3,3])
ax.set_ylim3d([-3,3])
ax.set_zlim3d([0,0.3])

plt.show()

# 2D visualization

fig, ax = plt.subplots(figsize=(8, 8))

ax.contour(XX,YY,pdf_fine,15,
           cmap = 'RdYlBu_r')
ax.axvline(x = 0, color = 'k', linestyle = '--')
ax.axhline(y = 0, color = 'k', linestyle = '--')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

def draw_vector(vector,RBG): 
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)

theta = np.arccos(rho)

fig, ax = plt.subplots()

draw_vector([1,0],np.array([0,112,192])/255)
draw_vector([np.cos(theta), np.sin(theta)],np.array([255,0,0])/255)

circle_theta = np.linspace(0, 2*np.pi, 100)

circle_X = np.cos(circle_theta)
circle_Y = np.sin(circle_theta)

ax.plot(circle_X, circle_Y, color = 'k', linestyle = '--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('scaled')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
