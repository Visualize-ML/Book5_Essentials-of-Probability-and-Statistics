
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import plotly.graph_objects as go # 导入 Plotly 的图形对象模块，用于创建交互式可视化
import numpy as np # 导入 NumPy，用于数值计算
import streamlit as st # 导入 Streamlit，用于创建交互式 Web 应用程序
from scipy.stats import multivariate_normal # 导入多变量正态分布模块

# 在 Streamlit 中显示数学公式，定义三维正态分布的概率密度函数
st.latex(r'''{\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})={\frac {\exp \left(-{\frac {1}{2}}({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right)}{\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}}''')

# 定义一个函数，将 NumPy 矩阵转换为 LaTeX bmatrix 格式
def bmatrix(a):
    """返回 LaTeX 格式的 bmatrix 表示

    :a: numpy 数组
    :returns: LaTeX 格式的 bmatrix 字符串
    """
    if len(a.shape) > 2: # 检查数组是否超过二维
        raise ValueError('bmatrix 最多支持两维数组') # 超出维度限制时报错
    lines = str(a).replace('[', '').replace(']', '').splitlines() # 格式化矩阵为字符串
    rv = [r'\begin{bmatrix}'] # 添加 bmatrix 开始语法
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines] # 按行格式化矩阵内容
    rv +=  [r'\end{bmatrix}'] # 添加 bmatrix 结束语法
    return '\n'.join(rv) # 返回 LaTeX 表示的字符串

# 创建三维网格 xxx1, xxx2, xxx3，步长为 0.2
xxx1, xxx2, xxx3 = np.mgrid[-3:3:0.2, -3:3:0.2, -3:3:0.2]

# 在侧边栏创建滑块，用于动态调整分布参数
with st.sidebar:
    sigma_1 = st.slider('sigma_1', min_value=0.5, max_value=3.0, value=1.0, step=0.1) # sigma_1 滑块
    sigma_2 = st.slider('sigma_2', min_value=0.5, max_value=3.0, value=1.0, step=0.1) # sigma_2 滑块
    sigma_3 = st.slider('sigma_3', min_value=0.5, max_value=3.0, value=1.0, step=0.1) # sigma_3 滑块

    rho_1_2 = st.slider('rho_1_2', min_value=-0.95, max_value=0.95, value=0.0, step=0.05) # rho_1_2 滑块
    rho_1_3 = st.slider('rho_1_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05) # rho_1_3 滑块
    rho_2_3 = st.slider('rho_2_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05) # rho_2_3 滑块

# 构造协方差矩阵 SIGMA
SIGMA = np.array([[sigma_1**2, rho_1_2*sigma_1*sigma_2, rho_1_3*sigma_1*sigma_3],
                  [rho_1_2*sigma_1*sigma_2, sigma_2**2, rho_2_3*sigma_2*sigma_3],
                  [rho_1_3*sigma_1*sigma_3, rho_2_3*sigma_2*sigma_3, sigma_3**2]])

# 在 Streamlit 中显示协方差矩阵的 LaTeX 表示
st.latex(r'\Sigma = ' + bmatrix(SIGMA))

# 定义均值向量 MU
MU = np.array([0, 0, 0])

# 将三维网格点合并为位置数组 pos，用于计算 PDF
pos = np.dstack((xxx1.ravel(), xxx2.ravel(), xxx3.ravel()))

# 创建多变量正态分布对象 rv
rv = multivariate_normal(MU, SIGMA)

# 计算概率密度函数值 PDF
PDF = rv.pdf(pos)

# 使用 Plotly 创建 3D 体积可视化
fig = go.Figure(data=go.Volume(
    x=xxx1.flatten(), # 展平的 x 坐标
    y=xxx2.flatten(), # 展平的 y 坐标
    z=xxx3.flatten(), # 展平的 z 坐标
    value=PDF.flatten(), # 展平的 PDF 值
    isomin=0, # 最小等值面
    isomax=PDF.max(), # 最大等值面
    colorscale='RdYlBu_r', # 使用红黄蓝颜色映射
    opacity=0.1, # 设置透明度
    surface_count=11, # 设置等值面数量
))

# 更新 3D 图的布局
fig.update_layout(scene=dict(
    xaxis_title=r'x_1', # x 轴标题
    yaxis_title=r'x_2', # y 轴标题
    zaxis_title=r'x_3'  # z 轴标题
), width=1000, margin=dict(r=20, b=10, l=10, t=10)) # 设置图表宽度和边距

# 在 Streamlit 中显示 Plotly 图
st.plotly_chart(fig, theme=None, use_container_width=True)
