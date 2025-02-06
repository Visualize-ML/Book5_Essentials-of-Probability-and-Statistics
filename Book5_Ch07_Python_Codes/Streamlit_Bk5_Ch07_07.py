
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############


import streamlit as st  # 导入Streamlit库，用于构建交互式Web应用程序
from scipy.stats import dirichlet  # 从SciPy库中导入Dirichlet分布相关函数
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import numpy as np  # 导入NumPy库，用于数值计算

x_array = np.linspace(0, 1, 200)  # 创建一个数组，包含[0, 1]范围内的200个点，用于绘图或计算

with st.sidebar:  # 在Streamlit应用程序的侧边栏中添加内容
    st.write('Dirichlet distribution PDF')  # 显示标题文字
    st.latex(r'''
             {\displaystyle f\left(x_{1},\ldots ,x_{K};\alpha _{1},\ldots ,\alpha _{K}\right)=
              {\frac {1}{\mathrm {B} ({\boldsymbol {\alpha }})}}\prod _{i=1}^{K}x_{i}^{\alpha _{i}-1}}''')  # 显示Dirichlet分布的PDF公式
             
    st.latex(r'''{\displaystyle \mathrm {B} ({\boldsymbol {\alpha }})=
             {\frac {\prod \limits _{i=1}^{K}\Gamma (\alpha _{i})}{\Gamma \left(\sum \limits _{i=1}^{K}\alpha _{i}\right)}},\qquad {\boldsymbol {\alpha }}=(\alpha _{1},\ldots ,\alpha _{K}).}''')  # 显示Dirichlet分布的归一化常数公式
    
    alpha_1_input = st.slider('alpha_1', min_value=1.0, max_value=10.0, value=2.0, step=0.1)  # 创建一个滑块，用于设置α_1参数
    alpha_2_input = st.slider('alpha_2', min_value=1.0, max_value=10.0, value=2.0, step=0.1)  # 创建一个滑块，用于设置α_2参数
    alpha_3_input = st.slider('alpha_3', min_value=1.0, max_value=10.0, value=2.0, step=0.1)  # 创建一个滑块，用于设置α_3参数

rv = dirichlet([alpha_1_input, alpha_2_input, alpha_3_input])  # 根据输入的α参数创建一个Dirichlet分布对象

x1 = np.linspace(0, 1, 201)  # 生成x_1的值范围，包含201个点
x2 = np.linspace(0, 1, 201)  # 生成x_2的值范围，包含201个点

xx1, xx2 = np.meshgrid(x1, x2)  # 创建网格点，用于在二维空间表示x_1和x_2的取值

xx3 = 1.0 - xx1 - xx2  # 计算x_3的值，满足Dirichlet分布的条件x_1 + x_2 + x_3 = 1
xx3 = np.where(xx3 > 0.0, xx3, np.nan)  # 如果x_3的值小于0，将其替换为NaN，避免无效计算

PDF_ff = rv.pdf(np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])))  # 计算每个网格点对应的Dirichlet分布PDF值
PDF_ff = np.reshape(PDF_ff, xx1.shape)  # 将PDF值重新形状化为与网格一致的形状

import plotly as py  # 导入Plotly库，用于高级绘图
import plotly.graph_objects as go  # 从Plotly库中导入图形对象
from plotly.subplots import make_subplots  # 导入子图创建函数

fig = make_subplots(rows=1, cols=1,
                    specs=[[{'is_3d': True}]])  # 创建一个3D子图，包含1行1列

fig.add_trace(go.Surface(x=xx1, y=xx2, z=xx3, surfacecolor=PDF_ff, colorscale='RdYlBu_r'), 1, 1)  
# 添加3D表面图，x和y为网格点，z为x_3，颜色根据PDF值设置，使用‘RdYlBu_r’颜色映射

fig.update_layout(scene=dict(
                    xaxis_title=r'x_1',  # 设置x轴标题为x_1
                    yaxis_title=r'x_2',  # 设置y轴标题为x_2
                    zaxis_title=r'x_3'),  # 设置z轴标题为x_3
                    width=700,  # 设置图形宽度
                    margin=dict(r=20, b=10, l=10, t=10))  # 设置图形边距

st.plotly_chart(fig, use_container_width=True)  # 在Streamlit应用程序中显示Plotly图表，并使图表适应容器宽度
