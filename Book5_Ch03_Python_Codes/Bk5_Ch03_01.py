
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt 

num_toss = 10000000
x = np.random.randint(1, 7, size = num_toss)
y = np.random.randint(1, 7, size = num_toss)

num_toss_array = np.arange(1, num_toss + 1)

sum_6 = np.cumsum((x + y) == 6); 

prob_sum_6 = sum_6/num_toss_array;

#%% Visualization

fig, ax = plt.subplots()

plt.plot(num_toss_array, prob_sum_6)

ax.set_xscale('log')
plt.xlabel('Number of tosses'); 
plt.ylabel('Probability')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
