

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt 

x = 2*np.random.uniform(0,1,1000000)
 
x_sq = x**2;
 
makers = (x_sq<=2);
 
est_sqrt_2 = 2*np.sum(makers)/len(x_sq);
err = (est_sqrt_2 - np.sqrt(2))/np.sqrt(2)*100; 
# percentage of error
