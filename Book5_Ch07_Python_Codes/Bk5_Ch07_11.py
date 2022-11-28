
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import scipy.stats as st
import scipy.interpolate as si
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Dirichlet PDF


# alpha = np.array([1, 1, 1])
# alpha = np.array([2, 2, 2])
# alpha = np.array([4, 4, 4])

# alpha = np.array([1, 4, 4])
# alpha = np.array([4, 1, 4])
# alpha = np.array([4, 4, 1])

# alpha = np.array([4, 2, 2])
# alpha = np.array([2, 4, 2])
# alpha = np.array([2, 2, 4])

# alpha = np.array([1, 2, 4])
# alpha = np.array([2, 1, 4])
alpha = np.array([4, 2, 1])

rv = st.dirichlet(alpha)

x1 = np.linspace(0,1,201)
x2 = np.linspace(0,1,201)

xx1, xx2 = np.meshgrid(x1, x2)

xx3 = 1.0 - xx1 - xx2
xx3 = np.where(xx3 > 0.0, xx3, np.nan)

PDF_ff = rv.pdf(np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])))
PDF_ff = np.reshape(PDF_ff, xx1.shape)

# PDF_ff = np.nan_to_num(PDF_ff)

#%% 2D contour

fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(xx1, xx2, PDF_ff, 20, cmap='RdYlBu_r')

#%% 3D contour

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx1, xx2, PDF_ff,
                  color = [0.7,0.7,0.7],
                  linewidth = 0.25,
                  rstride=10, cstride=10)

ax.contour(xx1, xx2, PDF_ff, 
           levels = 20,  cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,20])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,20])
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
ax.grid(True)
plt.show()

#%% 3D visualization

x1_ = np.linspace(0,1,51)
x2_ = np.linspace(0,1,51)

xx1_, xx2_ = np.meshgrid(x1_, x2_)

xx3_ = 1.0 - xx1_ - xx2_
xx3_ = np.where(xx3_ > 0.0, xx3_, np.nan)

PDF_ff_ = rv.pdf(np.array(([xx1_.ravel(), xx2_.ravel(), xx3_.ravel()])))
PDF_ff_ = np.reshape(PDF_ff_, xx1_.shape)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

# Creating plot
PDF_ff_ = np.nan_to_num(PDF_ff_)
ax.scatter3D(xx1_.ravel(), 
             xx2_.ravel(), 
             xx3_.ravel(), 
             c=PDF_ff_.ravel(), 
             marker='.',
             cmap = 'RdYlBu_r')

ax.contour(xx1_, xx2_, PDF_ff_, 15, zdir='z', offset=0, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_yticks(np.linspace(0,1,6))
# ax.set_zticks(np.linspace(0,1,6))

x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
u, v, w = np.array([[1.2,0,0],[0,1.2,0],[0,0,1.2]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
# ax.set_axis_off()

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,1])
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=-30, elev=20)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))

ax.grid()
plt.show()

#%% Marginal distributions

from scipy.stats import beta

x_array = np.linspace(0,1,200)

alpha_array = alpha
beta_array = alpha.sum() - alpha

# PDF of Beta Distributions

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

for alpha_idx, beta_idx, ax in zip(alpha_array.ravel(), beta_array.ravel(), axs.ravel()):


    title_idx = '\u03B1 = ' + str(alpha_idx) + '; \u03B2 = ' + str(beta_idx)
    ax.plot(x_array, beta.pdf(x_array, alpha_idx, beta_idx),
            lw=1)

    ax.set_xlim(0,1)
    ax.set_ylim(0,4)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,2,4])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')
    ax.set_box_aspect(1)
    ax.set_title(title_idx)

#%% Scatter plot of random data

random_data = np.random.dirichlet(alpha, 500).T

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

ax.scatter3D(random_data[0,:], 
             random_data[1,:], 
             random_data[2,:], 
             marker='.')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.set_xticks(np.linspace(0,1,6))
# ax.set_yticks(np.linspace(0,1,6))
# ax.set_zticks(np.linspace(0,1,6))

x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
u, v, w = np.array([[1.2,0,0],[0,1.2,0],[0,0,1.2]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
# ax.set_axis_off()

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,1])
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=30, elev=20)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))
ax.grid()
plt.show()
