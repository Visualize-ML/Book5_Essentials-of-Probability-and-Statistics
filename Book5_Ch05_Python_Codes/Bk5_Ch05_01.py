

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint

a = 1
b = 6

x = np.arange(a, b+1)

discrete_uniform = randint(a, b+1)

p_x = discrete_uniform.pmf(x)

E_x = np.sum(p_x*x)

fig, ax = plt.subplots()

plt.stem(x, p_x)
plt.axvline(x = E_x, color = 'r', linestyle = '--')

plt.xticks(np.arange(a,b+1))
plt.xlabel('x')
plt.ylabel('PMF, $p_X(x)$')
plt.ylim([0,0.2])
