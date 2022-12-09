

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
from sympy import symbols, ln, simplify, lambdify, diff, solve, Float
import matplotlib.pyplot as plt

theta_1, theta_2 = symbols('theta_1 theta_2')

samples = [-2.5, -5, 1, 3.5, -4, 1.5, 5.5]
mu = np.mean(samples)

print(mu)

n = len(samples)
bias_std = np.std(samples)
bias_var = bias_std**2
print(bias_var)

A = 0
for i in np.arange(n):
    term_i = (samples[i] - theta_1)**2
    A = A + term_i

A = simplify(A)
print(A)

lnL = -n/2*np.log(2*np.pi) - n/2*ln(theta_2) - 1/2/theta_2*A

#%%

lnL = simplify(lnL)

print(lnL)

theta_1_array = np.linspace(mu-3,mu+3,40)
theta_2_array = np.linspace(bias_var*0.8,bias_var*1.2,40)

theta_11,theta_22 = np.meshgrid(theta_1_array,theta_2_array)

lnL_fcn = lambdify((theta_1,theta_2), lnL)

lnL_matrix = lnL_fcn(theta_11,theta_22)

#%%
# first-order partial differential
df_dtheta_1 = diff(lnL, theta_1)
print(df_dtheta_1)

df_dtheta_2 = diff(lnL, theta_2)
print(df_dtheta_2)

# solution of (theta_1,theta_2)

sol = solve([df_dtheta_1, df_dtheta_2], [theta_1, theta_2])

print(sol)

theta_1_star = sol[0][0]
theta_2_star = sol[0][1]

theta_1_star = theta_1_star.evalf()
theta_2_star = str(theta_2_star)
theta_2_star = eval(theta_2_star)

print(theta_1_star)
print(theta_2_star)

lnL_min = lnL_fcn(theta_1_star,theta_2_star)
print(lnL_min)

#%%


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(theta_11, theta_22, lnL_matrix,
                  color = [0.5,0.5,0.5],
                  linewidth = 0.25)

plt.plot(theta_1_star, theta_2_star, lnL_min,
         marker = 'x', markersize = 12)

colorbar = ax.contour(theta_11, theta_22, lnL_matrix,30,
             cmap = 'RdYlBu_r')

fig.colorbar(colorbar, ax=ax)

ax.set_proj_type('ortho')

ax.set_xlabel('$\\theta_1$, $\\mu$')
ax.set_ylabel('$\\theta_2$, $\\sigma^2$')

plt.tight_layout()
ax.set_xlim(theta_11.min(), theta_11.max())
ax.set_ylim(theta_22.min(), theta_22.max())

ax.view_init(azim=-135, elev=30)

ax.grid(False)
plt.show()

fig, ax = plt.subplots()

colorbar = ax.contourf(theta_11, theta_22, lnL_matrix, 30, cmap='RdYlBu_r')
fig.colorbar(colorbar, ax=ax)
plt.plot(theta_1_star, theta_2_star, marker = 'x', markersize = 12)

ax.set_xlim(theta_11.min(), theta_11.max())
ax.set_ylim(theta_22.min(), theta_22.max())

ax.set_xlabel('$\\theta_1$, $\\mu$')
ax.set_ylabel('$\\theta_2$, $\\sigma^2$')
# plt.gca().set_aspect('equal', adjustable='box')

plt.show()
