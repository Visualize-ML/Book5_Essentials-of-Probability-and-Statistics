

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt

num = 2000
x1 = 2
x2 = 10
def f(x):
    return x*np.sin(x)/2 + 8

x = np.arange(x1, x2, 0.01)
y = f(x)
fmax = np.max(y)
x_rand = x1 + (x2 - x1)*np.random.random(num)
y_rand = np.random.random(num)*fmax

ind_below = np.where(y_rand < f(x_rand))
ind_above = np.where(y_rand >= f(x_rand))


fig, ax = plt.subplots()

plt.scatter(x_rand[ind_below], y_rand[ind_below], 
            color = "b", marker = '.')
plt.scatter(x_rand[ind_above], y_rand[ind_above], 
            color = "r", marker = 'x')
plt.plot(x, y, color = "k")
# plt.tight_layout()
plt.xlim(2,10)
plt.ylim(0,12)

estimated_area = np.sum(y_rand < f(x_rand))/num*fmax*(x2 - x1)
print(estimated_area)

from scipy.integrate import quad
integral = quad(f, x1, x2)
print(integral[0])
