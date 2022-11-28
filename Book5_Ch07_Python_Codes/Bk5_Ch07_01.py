

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a = 0
b = 10
num_data = 500

random_data = np.random.uniform(a,b,num_data)

fig, ax = plt.subplots()
# Plot the histogram
# sns.displot(random_data, bins=20)
sns.histplot(random_data, bins=20, ax = ax)
sns.rugplot(random_data, ax = ax)

plt.xlabel('x')
plt.ylabel('Count')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(a,b)
plt.xticks([0,2,4,6,8,10])


fig, ax = plt.subplots()
# Plot empirical cumulative distribution function
# sns.ecdfplot(random_data)

sns.histplot(random_data, bins=20, fill=True, 
             cumulative=True, stat="density")

plt.xlabel('x')
plt.ylabel('Empirical CDF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(a,b)
plt.xticks([0,2,4,6,8,10])
