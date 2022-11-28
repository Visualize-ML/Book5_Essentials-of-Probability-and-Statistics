

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.stats import hypergeom, binom
import matplotlib.pyplot as plt
import numpy as np

p = 0.3  # percentage of rabbits in the population

# N: total number of animals
for N in [100,200,400,800]:
    
    K = N*p  # number of rabbits among N
    n = 20   # number of draws without replacement
    
    hyper_g = hypergeom(N, K, n) 
    
    x_array = np.arange(np.maximum(0,n + K - N), np.minimum(K,n) + 1)
    
    pmf_binom   = binom.pmf(x_array, n, p)
    
    pmf_hyper_g = hyper_g.pmf(x_array)
    
    fig, ax = plt.subplots()
    
    plt.plot(x_array, pmf_hyper_g, '-bx')
    plt.plot(x_array, pmf_binom, '--rx')
    
    plt.xlabel('x')
    plt.ylabel('PMF')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xlim(x_array.min(),x_array.max())
    plt.ylim(0,0.225)
    plt.xticks([0,5,10,15])
