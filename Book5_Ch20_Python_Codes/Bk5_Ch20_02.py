

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# number of animal samples:
n = 200

# number of rabbits in data:
s = 60

# probability of rabbits among animals:
p = s/n
binom_dist = stats.binom(n, p)
mu = binom_dist.mean()
theta_array = np.linspace(0, 1, 500)

# prior distribution
# assumption: 1:1 ratio
# i.e., alpha = beta in Beta(alpha, beta) distribution

alpha_arrays = [1, 2, 8, 16, 32, 64]

for alpha in alpha_arrays:
    
    beta = alpha
    
    prior = stats.beta(alpha, beta)
    
    # posterior distribution
    posterior = stats.beta(s + alpha, n - s + beta)
    
    
    fig, ax = plt.subplots(figsize = (12,6))
    
    plt.plot(theta_array, prior.pdf(theta_array), 
             label='Prior', c='b')
    
    plt.plot(theta_array, posterior.pdf(theta_array), 
             label='Posterior', c='r')
    
    # factor_normalize = stats.binom(n, theta_array).pmf(s).sum()*1/500
    factor_normalize = 1/(n + 1)
    # note: multiplication factor normalize to 
    # normalize likelihood distribution
    plt.plot(theta_array, 
             stats.binom(n, theta_array).pmf(s)/factor_normalize, 
             label='Likelihood', c='g')
    
    # Prior mode
    try:
        plt.axvline((alpha-1)/(alpha+beta-2), 
                    c='b', linestyle='--', 
                    label='Prior mode')
    except:
        pass
    
    # MAP
    plt.axvline((s+alpha-1)/(n+alpha+beta-2), 
                c='r', linestyle='--', 
                label='MAP')
    
    # MLE
    plt.axvline(mu/n, 
                c='g', 
                linestyle='--', 
                label='MLE')
    
    plt.xlim([0, 1])
    plt.ylim([0, 15])
    plt.yticks([0,5,10,15])
    plt.xlabel(r'$\theta$')
    plt.ylabel('Density')
    plt.legend()

#%% 3D visualizations

#%% Prior distributions

a_list = np.arange(1,64 + 1)

theta_MAP = (s + a_list - 1)/(n + 2*a_list - 2)
theta_MLE = s/n

Prior_PDF_matrix = []

for a_idx in a_list:
    print(s + a_idx)
    print(n - s + a_idx)
    print('=================')
    posterior = stats.beta(a_idx, a_idx)
    pdf_idx = posterior.pdf(theta_array)
    Prior_PDF_matrix.append(pdf_idx)

Prior_PDF_matrix = np.array(Prior_PDF_matrix)


fig, ax = plt.subplots(figsize=(10,10))
plt.contourf(theta_array, a_list, Prior_PDF_matrix, 
             levels = np.linspace(0,Prior_PDF_matrix.max()*1.2,10),
             cmap = 'Blues')

# plt.plot(theta_MAP,a, color = 'k')
# plt.axvline(x = theta_MLE, color = 'k')
# prior mode
plt.axvline(x = 0.5, color = 'k')

plt.xlabel('Theta')
plt.ylabel('Alpha')
plt.xlim(0,1)
plt.ylim(1,a_list.max())
plt.yticks([1,2,8, 16, 32, 64])


fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection': '3d'})

tt,aa = np.meshgrid(theta_array,a_list)
ax.plot_wireframe(tt, aa, Prior_PDF_matrix,
                  color = [0,0,0],
                  linewidth = 0.25,
                  rstride=3, cstride=0)

ax.contour(theta_array,a_list, Prior_PDF_matrix, 
             levels = np.linspace(0,Prior_PDF_matrix.max()*1.2,10),
             cmap = 'Blues')

ax.set_proj_type('ortho')

plt.xlabel('Theta')
plt.ylabel('Alpha')
plt.xlim(0,1)
plt.ylim(1,a_list.max())
plt.yticks([1,2,8, 16, 32, 64])

ax.set_zlim3d([0,30])
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
ax.grid(False)

#%% Posterior distribution

Prior_PDF_matrix = []

for a_idx in a_list:
    print(s + a_idx)
    print(n - s + a_idx)
    print('=================')
    posterior = stats.beta(s + a_idx, n - s + a_idx)
    pdf_idx = posterior.pdf(theta_array)
    Prior_PDF_matrix.append(pdf_idx)

Prior_PDF_matrix = np.array(Prior_PDF_matrix)


fig, ax = plt.subplots(figsize=(10,10))
plt.contourf(theta_array, a_list, Prior_PDF_matrix, 
             levels = np.linspace(0,Prior_PDF_matrix.max()*1.2,10),
             cmap = 'Blues')

plt.plot(theta_MAP,a_list, color = 'k')
plt.axvline(x = theta_MLE, color = 'k')
# prior mode
plt.axvline(x = 0.5, color = 'k')

plt.xlabel('Theta')
plt.ylabel('Alpha')
plt.xlim(0,1)
plt.ylim(1,a_list.max())
plt.yticks([1,2,8, 16, 32, 64])


fig, ax = plt.subplots(figsize=(10,10),
                       subplot_kw={'projection': '3d'})

tt,aa = np.meshgrid(theta_array,a_list)
ax.plot_wireframe(tt, aa, Prior_PDF_matrix,
                  color = [0,0,0],
                  linewidth = 0.25,
                  rstride=3, cstride=0)

ax.contour(theta_array, a_list, Prior_PDF_matrix, 
             levels = np.linspace(0,Prior_PDF_matrix.max()*1.2,10),
             cmap = 'Blues')

ax.set_proj_type('ortho')

plt.xlabel('Theta')
plt.ylabel('Alpha')
plt.xlim(0,1)
plt.ylim(1,a_list.max())
plt.yticks([1,2,8, 16, 32, 64])

ax.set_zlim3d([0,30])
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
ax.grid(False)
# plt.show()
