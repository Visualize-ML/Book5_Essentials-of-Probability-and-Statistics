

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps

def uni_normal_pdf(x,mu,sigma):
    
    coeff = 1/np.sqrt(2*np.pi)/sigma
    z = (x - mu)/sigma
    f_x = coeff*np.exp(-1/2*z**2)
    
    return f_x

X  = np.arange(-5,5,0.05)
mu = np.arange(-2,2,0.2)

XX, MM = np.meshgrid(X, mu)

sigma = 1

f_x_varying_mu = uni_normal_pdf(XX,MM,sigma)

# surface projected along Y to X-Z plane

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, MM, f_x_varying_mu, rstride=1, cstride=0,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour(XX, MM, f_x_varying_mu, 
           levels = 20, zdir='y', \
            offset=mu.max(), cmap=cm.RdYlBu_r)

ax.set_xlabel('$X$')
ax.set_ylabel('$\mu$')
ax.set_zlabel('$f_{X}(x)$')

ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)

ax.view_init(azim=-120, elev=30)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(mu)))

for i in np.linspace(1,len(mu),len(mu)):
    plt.plot(X,f_x_varying_mu[int(i)-1,:],
             color = colors[int(i)-1])

plt.xlabel('X')
plt.ylabel('$f_{X}(x)$')
ax.set_xlim(-5, 5)
ax.set_ylim(0, f_x_varying_mu.max())
