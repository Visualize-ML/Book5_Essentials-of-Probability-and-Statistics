

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt

X = np.random.uniform(-1, 1, size = (500,2))
x = X[:,0]
y = X[:,1]

masks = np.sqrt(x**2 + y**2) < 1

pi_est = 4 * sum(masks)/len(x)

fig, ax = plt.subplots()

# plot a unit circle
circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None')
ax.add_patch(circ)

# plot data inside the circle
plt.scatter(x[masks], y[masks], marker="x", alpha=0.5, color = 'b')

# plot data outside the circle
plt.scatter(x[~masks], y[~masks], marker=".", alpha=0.5, color = 'r')
plt.axis('scaled')
plt.title('Estimated $\pi$ = %1.3f' %(pi_est))
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# define a function of estimating pi

def est_pi(n):
    X = np.random.uniform(-1, 1, size = (int(n),2))
    x = X[:,0]
    y = X[:,1]
    
    masks = np.sqrt(x**2 + y**2) < 1
    
    pi_est = 4 * sum(masks)/len(x)
    
    return pi_est

n_array = np.linspace(1000,1000*100,100)

est_pi_array = np.empty(len(n_array))

# convergence of estimated pi

i = 0
for n in n_array:
    pi_est = est_pi(n)
    est_pi_array[i] = pi_est
    i = i + 1

fig, ax = plt.subplots()

plt.semilogx(n_array, est_pi_array)
plt.xlabel("Number of random number")
plt.ylabel("Estimated $\pi$")
plt.axhline(np.pi, color="r");

plt.grid(True, which="both", ls="--")
