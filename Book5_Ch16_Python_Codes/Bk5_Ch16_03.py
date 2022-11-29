

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def f(x,y):
    return 2 - x**2 - y**2

x = np.arange(-1, 1, 0.1)
[X,Y] = np.meshgrid(x,x);
 
Z = f(X,Y)
 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot wireframe.
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

num_rnd = 5000;
x_rand = 2*np.random.random(num_rnd) - 1;
y_rand = 2*np.random.random(num_rnd) - 1;
z_rand = 2*np.random.random(num_rnd);

ind_below = np.where(z_rand < f(x_rand,y_rand))
ind_above = np.where(z_rand >= f(x_rand,y_rand))
 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter3D(x_rand[ind_below], y_rand[ind_below], z_rand[ind_below],
            color = "b", marker = '.')
ax.scatter3D(x_rand[ind_above], y_rand[ind_above], z_rand[ind_above],
            color = "r", marker = '.')

# Plot wireframe.
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

estimated_volume = np.sum(z_rand < f(x_rand,y_rand))/num_rnd*8
print(estimated_volume)

from scipy import integrate

double_integral = integrate.dblquad(f, -1, 1, lambda x: -1, lambda x: 1)
print(double_integral[0])
