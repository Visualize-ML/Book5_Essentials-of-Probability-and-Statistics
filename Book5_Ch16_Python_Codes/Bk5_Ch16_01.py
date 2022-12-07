

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.arange(-10, 10+0.1, 0.1)

# Distance of random walk at t
mu = 0;
# W(0) = 0, which is the starting point of random walk

t = np.arange(1,21)

sigma = 1; 
# standard normal distribution
sigma_series = np.sqrt(t)*sigma;

xx,tt = np.meshgrid(x,t);
 
fig, ax = plt.subplots()
plt.plot(t,sigma_series,marker = 'x')
plt.xlabel('t'); 
plt.ylabel('Sigma*sqrt(t)')
plt.xlim (t.min(),t.max())
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.xticks([1,5,10,15,20])


n_sigma = len(sigma_series);

colors = plt.cm.jet(np.linspace(0,1,n_sigma))
colors = np.flipud(colors)

fig, ax = plt.subplots()

pdf_matrix = [];

for i, t_i in zip(range(n_sigma),t):
    
    sigma = sigma_series[i];
    norm_ = norm(mu, sigma)
    pdf_x = norm_.pdf(x)
    
    plt.plot(x, pdf_x, color=colors[i], label='t = %s' % t_i)
    pdf_matrix.append(pdf_x)

plt.xlabel('x')
plt.ylabel('Probability density')
plt.legend()
pdf_matrix = np.array(pdf_matrix)
