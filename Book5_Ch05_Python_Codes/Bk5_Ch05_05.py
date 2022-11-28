

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

x_array = np.arange(0,20 + 1)


# PMF versus x as lambda varies

fig, ax = plt.subplots()

for lambda_ in [1,2,3,4,5,6,7,8,9,10]:
    plt.plot(x_array, poisson.pmf(x_array, lambda_), 
             marker = 'x',markersize = 8,
             label = '$\lambda$ = ' + str(lambda_))

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(0,x_array.max())
plt.ylim(0,0.4)
plt.xticks([0,5,10,15,20])
plt.legend()
