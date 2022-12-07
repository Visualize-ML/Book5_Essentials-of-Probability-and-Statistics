

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N_steps = 200; 
# number of steps

mu_1 = 0.2
mu_2 = 0.3

mu = np.matrix([mu_1, mu_2])

sigma_1 = 1
sigma_2 = 2
rho_1_2 = 0.8
delta_t = 1

SIGMA = np.matrix([[sigma_1**2, sigma_1*sigma_2*rho_1_2],
                   [sigma_1*sigma_2*rho_1_2, sigma_2**2]])

L = np.linalg.cholesky(SIGMA)
R = L.T

Z = np.random.normal(size = (N_steps,2))

delta_X = mu*delta_t + Z@R*np.sqrt(delta_t)

X = np.cumsum(delta_X, axis = 0); 

X_0 = np.zeros((1,2))
X = np.vstack((X_0,X))

t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t

rows = 1
cols = 2

fig, ax= plt.subplots()

ax.plot(t_n, X, lw=1)
ax.plot(t_n, mu_1*t_n,color = 'r', lw=0.25)
ax.plot(t_n, mu_2*t_n,color = 'r', lw=0.25)

ax.set_xlim([0,N_steps])
# ax.set_ylim([-20,100])
# ax.set_yticks([-20, 0, 20, 40, 60, 80])
ax.set_xlabel('t')
#%% scatter plot

import pandas as pd

delta_X_df = pd.DataFrame(data=delta_X, columns=["Delta x1", "Delta x2"])

fig, ax= plt.subplots()

# ax.scatter([delta_X[:,0]],[delta_X[:,1]])
sns.jointplot(data = delta_X_df, x = "Delta x1", y = "Delta x2")
