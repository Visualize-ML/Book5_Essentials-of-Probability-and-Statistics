

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

KK = [1,2,3,4,8,16,32,64]
p = 0.7 # 0.5

for K in KK:
    
    x = np.arange(0, K + 1)
    
    p_x= binom.pmf(x, K, p)
    
    E_x = np.sum(p_x*x)
    
    fig, ax = plt.subplots()
    plt.stem(x, p_x)
    plt.axvline(x = E_x, color = 'r', linestyle = '--')
    
    plt.xticks(np.arange(K+1))
    plt.xlabel('X = x')
    plt.ylabel('PMF, $p_X(x)$')
    plt.ylim([0,p +0.05])
