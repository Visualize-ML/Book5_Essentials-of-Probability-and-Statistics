
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import streamlit as st


from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import numpy as np

x_array = np.linspace(0,1,200)

with st.sidebar:
    st.write('Dirichlet distribution PDF')
    st.latex(r'''
             {\displaystyle f\left(x_{1},\ldots ,x_{K};\alpha _{1},\ldots ,\alpha _{K}\right)=
              {\frac {1}{\mathrm {B} ({\boldsymbol {\alpha }})}}\prod _{i=1}^{K}x_{i}^{\alpha _{i}-1}}''')
             
    st.latex(r'''{\displaystyle \mathrm {B} ({\boldsymbol {\alpha }})=
             {\frac {\prod \limits _{i=1}^{K}\Gamma (\alpha _{i})}{\Gamma \left(\sum \limits _{i=1}^{K}\alpha _{i}\right)}},\qquad {\boldsymbol {\alpha }}=(\alpha _{1},\ldots ,\alpha _{K}).}''')
             
    alpha_1_input = st.slider('alpha_1', min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    alpha_2_input = st.slider('alpha_2', min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    alpha_3_input = st.slider('alpha_3', min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    
    
rv = dirichlet([alpha_1_input, alpha_2_input, alpha_3_input])

x1 = np.linspace(0,1,201)
x2 = np.linspace(0,1,201)

xx1, xx2 = np.meshgrid(x1, x2)

xx3 = 1.0 - xx1 - xx2
xx3 = np.where(xx3 > 0.0, xx3, np.nan)

PDF_ff = rv.pdf(np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])))
PDF_ff = np.reshape(PDF_ff, xx1.shape)


import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=1,
                    specs=[[{'is_3d': True}]])

fig.add_trace(go.Surface(x=xx1, y=xx2, z=xx3, surfacecolor=PDF_ff, colorscale='RdYlBu_r'), 1, 1)

fig.update_layout(scene = dict(
                    xaxis_title=r'x_1',
                    yaxis_title=r'x_2',
                    zaxis_title=r'x_3'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))

st.plotly_chart(fig, use_container_width=True)