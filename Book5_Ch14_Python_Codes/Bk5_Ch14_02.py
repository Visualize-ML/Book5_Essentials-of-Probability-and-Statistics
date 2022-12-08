
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

mean = [1, 2]
# center of data
cov = [[1, 1], [1, 1.5]]  
# covariance matrix

X = np.random.multivariate_normal(mean, cov, 500)

fig, ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')

plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])

X_df = pd.DataFrame(X, columns=['x_1', 'x_2'])

sns.jointplot(data=X_df,x = 'x_1', y = 'x_2', 
              kind = 'kde', fill = True,
              xlim = (-3,5), ylim = (-2,6))

ax.set_aspect('equal')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')


#%% PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

fig, ax = plt.subplots()

plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')

plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

# plot first principal component, PC1

PC1_x = pca.components_[0,0]
PC1_y = pca.components_[0,1]

ax.quiver(mean[0],mean[1],PC1_x,PC1_y,
          angles='xy', scale_units='xy',scale=1/3, 
          edgecolor='none', facecolor= 'b')

# plot second principal component, PC2

PC2_x = pca.components_[1,0]
PC2_y = pca.components_[1,1]

ax.quiver(mean[0],mean[1],PC2_x,PC2_y,
          angles='xy', scale_units='xy',scale=1/3, 
          edgecolor='none', facecolor= 'r')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])

# convert X to Z

Z = pca.transform(X)

Z_df = pd.DataFrame(Z, columns=['z_1', 'z_2'])

fig, ax = plt.subplots()

sns.kdeplot(data=Z_df)
sns.rugplot(data=Z_df)

fig, ax = plt.subplots()
plt.scatter(Z[:, 0], Z[:, 1], alpha = 0.5, marker = '.')

plt.axvline(x=0, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')

plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.axis('scaled')

ax.set_xlim([-5,5])
ax.set_ylim([-5,5])

sns.jointplot(data=Z_df,x = 'z_1', y = 'z_2', 
              kind = 'kde', fill = True,
              xlim = (-5,5), ylim = (-5,5))

ax.set_aspect('equal')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')

#%% dimension reduction

pca_PC1 = PCA(n_components=1)
pca_PC1.fit(X)

z1 = pca_PC1.transform(X)

x1_proj = pca_PC1.inverse_transform(z1)

fig, ax = plt.subplots()

plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')

# plot first principal component, PC1

PC1_x = pca_PC1.components_[0,0]
PC1_y = pca_PC1.components_[0,1]

ax.quiver(mean[0],mean[1],PC1_x,PC1_y,
          angles='xy', scale_units='xy',scale=1/3, 
          edgecolor='none', facecolor= 'b')

plt.scatter(x1_proj[:, 0], x1_proj[:, 1], alpha=0.5, c = 'k', marker = 'x')


plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])
