

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm # Colormaps
import streamlit as st

with st.sidebar:
    rho = st.slider('rho', min_value = -0.95, max_value = 0.95, value = 0.0, step = 0.05)
    sigma_X = st.slider('sigma_X', min_value = 0.5, max_value = 3.0, value = 1.0, step = 0.1)
    sigma_Y = st.slider('sigma_Y', min_value = 0.5, max_value = 3.0, value = 1.0, step = 0.1)

mu_X = 0
mu_Y = 0

st.write('Bivariate Gaussian distribution PDF')
st.latex(r'''{\displaystyle f(x,y)={\frac {1}{2\pi \sigma _{X}\sigma _{Y}{\sqrt {1-\rho ^{2}}}}}\exp \left(-{\frac {1}{2(1-\rho ^{2})}}\left[\left({\frac {x-\mu _{X}}{\sigma _{X}}}\right)^{2}-2\rho \left({\frac {x-\mu _{X}}{\sigma _{X}}}\right)\left({\frac {y-\mu _{Y}}{\sigma _{Y}}}\right)+\left({\frac {y-\mu _{Y}}{\sigma _{Y}}}\right)^{2}\right]\right)}''')
   

mu    = [mu_X, mu_Y]
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], 
        [sigma_X*sigma_Y*rho, sigma_Y**2]]

width = 4
X = np.linspace(-width,width,321)
Y = np.linspace(-width,width,321)

XX, YY = np.meshgrid(X, Y)

XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(mu, Sigma)

#%% visualize joint PDF surface

f_X_Y_joint = bi_norm.pdf(XXYY)

# 3D visualization

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(XX,YY, f_X_Y_joint,
                  rstride=10, cstride=10,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

ax.contour3D(XX,YY, f_X_Y_joint,15,
             cmap = 'RdYlBu_r')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{X,Y}(x,y)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
ax.view_init(azim=-120, elev=30)
plt.tight_layout()

st.pyplot(fig)


#%% Plot filled contours

fig, ax = plt.subplots(figsize=(7, 7))

# Plot bivariate normal
plt.contourf(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
plt.axvline(x = mu_X, color = 'k', linestyle = '--')
plt.axhline(y = mu_Y, color = 'k', linestyle = '--')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

st.pyplot(fig)
