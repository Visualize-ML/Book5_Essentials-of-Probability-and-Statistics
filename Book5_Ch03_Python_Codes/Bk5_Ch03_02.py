

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt 
 
num_toss = 10000;
toss_times = 10;
 
toss_results = np.random.randint(0, 2, size = (num_toss, toss_times))

sum_results = np.sum(toss_results,axis = 1);

result_exact_6 = (sum_results == 6);

result_at_least_6 = (sum_results >= 6);
 
num_toss_array = np.arange(1, num_toss + 1)

count_6 = np.cumsum(result_exact_6);
count_6_at_least = np.cumsum(result_at_least_6);

prob_count_6 = count_6/num_toss_array;
prob_count_6_at_least = count_6_at_least/num_toss_array;

#%% Visualization

fig, ax = plt.subplots()

plt.plot(num_toss_array, prob_count_6)

ax.set_xscale('log')
plt.xlabel('Number of tosses'); 
plt.ylabel('Probability')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

fig, ax = plt.subplots()

plt.plot(num_toss_array, prob_count_6_at_least)

ax.set_xscale('log')
plt.xlabel('Number of tosses'); 
plt.ylabel('Probability')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
