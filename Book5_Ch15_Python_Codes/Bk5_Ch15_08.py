

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mu_z    = [0, 0]
SIGMA_z = [[1, 0], [0, 1]]  
# diagonal covariance

z1, z2 = np.random.multivariate_normal(mu_z, SIGMA_z, 500).T
Z       = np.matrix([z1,z2]).T
# IID standard normal


Z_df = pd.DataFrame(data=Z, columns=["Z1", "Z2"])

g = sns.jointplot(data = Z_df, x='Z1', y='Z2', 
                  alpha = 0.5, color = 'b', 
                  xlim = (-4,8), ylim = (-4,8))

g.plot_joint(sns.kdeplot, color="g", zorder=0, fill = False)
g.plot_marginals(sns.rugplot, color="k")

g.ax_joint.axvline(x=Z_df.mean()['Z1'], color = 'r', linestyle = '--')
g.ax_joint.axhline(y=Z_df.mean()['Z2'], color = 'r', linestyle = '--')

#%% Use Cholesky decomposition
#   generate multivariate normal random numbers

E_X     = [2, 4]
SIGMA_X = [[4, 2], [2, 2]]

# x1, x2 = np.random.multivariate_normal(E_x, SIGMA_x, 500).T

L = np.linalg.cholesky(SIGMA_X)
R = L.T 

X_Chol = Z@R + np.matrix([E_X])

X_Chol_df = pd.DataFrame(data=X_Chol, columns=["X1", "X2"])

g = sns.jointplot(data = X_Chol_df, x='X1', y='X2', 
                  alpha = 0.5, color = 'b',
                  xlim = (-4,8), ylim = (-4,8))

g.plot_joint(sns.kdeplot, color="g", zorder=0, fill = False)
g.plot_marginals(sns.rugplot, color="k")
g.ax_joint.axvline(x=X_Chol_df.mean()['X1'], color = 'r', linestyle = '--')
g.ax_joint.axhline(y=X_Chol_df.mean()['X2'], color = 'r', linestyle = '--')
