
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

with st.sidebar:
    theta = st.slider('theta (degrees)', min_value = 0, 
                        max_value = 180, 
                        step = 10,
                        value = 0)
    
    theta = theta/180 * np.pi


num_points = 37;
theta_array = np.linspace(0, 2*np.pi, num_points).reshape((-1, 1))


X_circle = np.column_stack((np.cos(theta_array),
                            np.sin(theta_array)))

colors = plt.cm.rainbow(np.linspace(0,1,len(X_circle)))

theta_array_fine = np.linspace(0, 2*np.pi, 500).reshape((-1, 1))


X_circle_fine = np.column_stack((np.cos(theta_array_fine),
                            np.sin(theta_array_fine)))


X_square = np.array([[0, 0],
                     [0, 1],
                     [1, 1],
                     [1, 0],
                     [0, 0]])


X_square_big = np.array([[1, 1],
                         [1, -1],
                         [-1, -1],
                         [-1, 1],
                         [1, 1]])

center_array = X_circle*0

A = np.array([[1.25, -0.75],
              [-0.75,1.25]])

SIGMA = A.T @ A
L = np.linalg.cholesky(SIGMA)
R = L.T 

fig, ax = plt.subplots(figsize = (10,10))


U = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])


X_square_R_rotated = X_square @ U
X_square_big_R_rotated = X_square_big @ U


X_R = X_circle @ U @ R
X_circle_fine_R = X_circle_fine @ U @ R
X_square_R = X_square @ U @ R
X_square_big_R = X_square_big @ U @R



ax.plot(X_circle_fine[:,0],X_circle_fine[:,1],c=[0.8,0.8,0.8], lw = 0.2)


ax.scatter(X_circle[:,0],X_circle[:,1], s = 200, 
           marker = '.', c = colors, zorder=1e3)


ax.plot(X_square[:,0],X_square[:,1],c='k', linewidth = 1)
ax.plot(X_square_big[:,0],X_square_big[:,1],c='k')


ax.plot(X_square_R[:,0],X_square_R[:,1],c='k', linewidth = 1)
ax.plot(X_square_big_R[:,0],X_square_big_R[:,1],c='k')


ax.plot(X_square_R_rotated[:,0],X_square_R_rotated[:,1],c='k', linewidth = 1)
ax.plot(X_square_big_R_rotated[:,0],X_square_big_R_rotated[:,1],c='k')


ax.plot(X_circle_fine_R[:,0],X_circle_fine_R[:,1],c=[0.8,0.8,0.8], lw = 0.2)


ax.plot(([i for (i,j) in X_circle], [i for (i,j) in X_R]),
          ([j for (i,j) in X_circle], [j for (i,j) in X_R]),c=[0.8,0.8,0.8], lw = 0.2)


ax.scatter(X_R[:,0],X_R[:,1], s = 200, 
           marker = '.', c = colors, zorder=1e3)


ax.axvline(x = 0, c = 'k', lw = 0.2)
ax.axhline(y = 0, c = 'k', lw = 0.2)


ax.axis('scaled')
ax.set_xbound(lower = -2.5, upper = 2.5)
ax.set_ybound(lower = -2.5, upper = 2.5)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

st.pyplot(fig)


