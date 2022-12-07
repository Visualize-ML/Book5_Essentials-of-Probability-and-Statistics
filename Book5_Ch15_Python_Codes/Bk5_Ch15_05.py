

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt

l = 1
# length of needles
t = 2
# distance between parallel lines

num = 2000
theta_1 = 0
theta_2 = np.pi/2

def f(theta):
    return np.sin(theta)/2

theta_array = np.arange(theta_1, theta_2, np.pi/100)
x_array = f(theta_array)
x_max = t/2
theta_rand = theta_1 + (theta_2 - theta_1)*np.random.random(num)
x_rand = np.random.random(num)*x_max

ind_below = np.where(x_rand < f(theta_rand))
ind_above = np.where(x_rand >= f(theta_rand))


fig, ax = plt.subplots()

plt.scatter(theta_rand[ind_below], x_rand[ind_below], 
            color = "b", marker = '.')
plt.scatter(theta_rand[ind_above], x_rand[ind_above], 
            color = "r", marker = 'x')
plt.plot(theta_array, x_array, color = "k")
plt.tight_layout()
plt.xlabel('$\u03B8$')
plt.ylabel('$x$')

estimated_pi = num/np.sum(x_rand < f(theta_rand))*2*l/t
print(estimated_pi)
