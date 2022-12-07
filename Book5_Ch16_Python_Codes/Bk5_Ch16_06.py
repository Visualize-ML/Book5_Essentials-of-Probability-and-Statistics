

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
N_paths = 500;
# number of paths
sigma = 0.2
delta_t = 1
mu = 0.005
sigma = 0.05

X_0 = 10

t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t

X = np.exp(
    (mu - sigma ** 2 / 2) * delta_t
    + sigma * np.random.normal(0, np.sqrt(delta_t), size=(N_steps,N_paths)))

X = np.vstack([np.ones((1,N_paths)), X])
X = X_0 * X.cumprod(axis=0)

rows = 1
cols = 2

fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=(10,5), gridspec_kw={'width_ratios': [3, 1]})

ax1.plot(t_n, X, lw=0.25,color = '#0070C0')
ax1.set_xlim([0,N_steps])
ax1.set_ylim([-25,200])
ax1.set_yticks([-20, 0, 50, 100, 150, 200])
ax1.set_title('(a)', loc='left')
ax1.set_xlabel('t')

ax2 = sns.distplot(X[-1], rug=True, rug_kws={"color": "k", 
                                                "alpha": 0.5, 
                                                "height": 0.06, 
                                                "lw": 0.5}, 
                   vertical=True, label='(b)', bins = 20)
ax2.set_yticks([-20, 0, 50, 100, 150, 200])
ax2.set_title('(b)', loc='left')
ax2.set_ylim([-25,200])
