

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt

N_steps = 1000; 
# number of steps

delta_x = np.random.normal(loc=0.0, scale=1.0, size=(N_steps,1))
delta_y = np.random.normal(loc=0.0, scale=1.0, size=(N_steps,1))
 
disp_x = np.cumsum(delta_x, axis = 0); 
disp_y = np.cumsum(delta_y, axis = 0); 

disp_x = np.vstack(([0],disp_x))
disp_y = np.vstack(([0],disp_y))


fig, ax = plt.subplots()

plt.plot(disp_x,disp_y); 
plt.plot(0,0,'rx')
plt.ylabel('$x$'); 
plt.xlabel('$y$'); 
plt.axis('equal')

#%% Snapshots at various time stamps

fig, axs = plt.subplots(1, 5, figsize=(20,4))

for i in np.linspace(0,4,5):
    i = int(i)
    X_i = X[int(i + 1)*40]
    E_X_i = X_i.mean()
    std_X_i = X_i.std()
    
    sns.distplot(X_i,rug=True, ax = axs[i],bins = 15,
                 hist_kws=dict(edgecolor="b", linewidth=0.25),
                 rug_kws={"color": "k", "alpha": 0.5, 
                          "height": 0.06, "lw": 0.5})
    
    axs[i].plot(E_X_i, 0, 'xr')
    axs[i].axvline(x = E_X_i, color = 'r', ymax = 0.9)
    axs[i].axvline(x = E_X_i + std_X_i, color = 'r', ymax = 0.7)
    axs[i].axvline(x = E_X_i - std_X_i, color = 'r', ymax = 0.7)
    axs[i].axvline(x = E_X_i + 2*std_X_i, color = 'r', ymax = 0.5)
    axs[i].axvline(x = E_X_i - 2*std_X_i, color = 'r', ymax = 0.5)
    axs[i].set_xticks([-50, 0, 50])
    axs[i].set_xlim([-60,60])
    axs[i].set_ylim([0,0.08])
