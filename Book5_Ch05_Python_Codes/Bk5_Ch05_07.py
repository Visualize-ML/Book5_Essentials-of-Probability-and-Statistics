

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import numpy as np

N = 50 # total number of animals
K = 15 # number of rabbits among N
n = 20 # number of draws without replacement

hyper_g = hypergeom(N, K, n) 

x_array = np.arange(np.maximum(0,n + K - N), np.minimum(K,n) + 1)

pmf_rabbits = hyper_g.pmf(x_array)

fig, ax = plt.subplots()

plt.stem(x_array, pmf_rabbits)

plt.xlabel('x')
plt.ylabel('PMF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(x_array.min(),x_array.max())
plt.ylim(0,pmf_rabbits.max())
plt.xticks([0,5,10,15])
