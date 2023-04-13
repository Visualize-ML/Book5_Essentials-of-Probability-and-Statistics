
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import streamlit as st


from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

x_array = np.linspace(0,1,200)

with st.sidebar:
    st.write('Beta distribution PDF')
    st.latex(r'''
             {\displaystyle {\begin{aligned}f(x;\alpha ,\beta )&=
                             \mathrm {constant} \cdot x^{\alpha -1}(1-x)^{\beta -1}\\
                                 &=
                             {\frac {x^{\alpha -1}(1-x)^{\beta -1}}
                              {\displaystyle \int _{0}^{1}u^{\alpha -1}
                               (1-u)^{\beta -1}\,du}}\\
                                 &=
                             {\frac {\Gamma (\alpha +\beta )}
                              {\Gamma (\alpha )\Gamma (\beta )}}\,x^{\alpha -1}(1-x)^{\beta -1}\\
                                 &=
                                 {\frac {1}{\mathrm {B} (\alpha ,\beta )}}
                                 x^{\alpha -1}(1-x)^{\beta -1}\end{aligned}}}
             ''')
             
    alpha_input = st.slider('alpha', min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    beta_input  = st.slider('beta', min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    
    
mean_loc  = alpha_input/(alpha_input + beta_input)
pdf_array = beta.pdf(x_array, alpha_input, beta_input)


fig, ax = plt.subplots(figsize=(8, 8))

title_idx = '\u03B1 = ' + str(alpha_input) + '; \u03B2 = ' + str(beta_input)
ax.plot(x_array, pdf_array,
        'b', lw=1)

ax.axvline (x = mean_loc, c = 'r', ls = '--')
ax.set_title(title_idx)
ax.set_xlim(0,1)
ax.set_ylim(0,4)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,2,4])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis="x", direction='in')
ax.tick_params(axis="y", direction='in')

st.pyplot(fig)