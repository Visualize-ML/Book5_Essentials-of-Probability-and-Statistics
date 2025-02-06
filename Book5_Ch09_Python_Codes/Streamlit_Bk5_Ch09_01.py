
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import streamlit as st  # 导入Streamlit库，用于创建交互式Web应用程序
from scipy.stats import beta  # 从SciPy库导入Beta分布相关函数（此代码未使用）
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import numpy as np  # 导入NumPy库，用于数值计算

def uni_normal_pdf(x, mu, sigma):  # 定义一个函数，用于计算一元正态分布的概率密度函数
    coeff = 1 / np.sqrt(2 * np.pi) / sigma  # 计算正态分布的系数部分 1/(σ√(2π))
    z = (x - mu) / sigma  # 计算标准化后的值 (x-μ)/σ
    f_x = coeff * np.exp(-1 / 2 * z**2)  # 计算概率密度值 f(x) = coeff * exp(-1/2 * z^2)
    return f_x  # 返回概率密度值

x_array = np.linspace(-5, 5, 200)  # 创建一个包含[-5, 5]范围内200个点的数组，用于绘制PDF

with st.sidebar:  # 在Streamlit的侧边栏中添加控件
    mu_input = st.slider('mu', min_value=-5.0, max_value=5.0, value=0.0, step=0.2)  
    # 添加滑块控件，用于调整正态分布的均值μ，初始值为0.0，范围[-5, 5]，步长为0.2
    sigma_input = st.slider('sigma', min_value=0.0, max_value=4.0, value=1.0, step=0.1)  
    # 添加滑块控件，用于调整正态分布的标准差σ，初始值为1.0，范围[0, 4]，步长为0.1

st.write('Univariate Gaussian distribution PDF')  # 在主页面显示正态分布的标题
st.latex(r'''{\displaystyle f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x-\mu }{\sigma }}\right)^{2}}}''')  
# 在主页面显示正态分布的LaTeX公式

pdf_array = uni_normal_pdf(x_array, mu_input, sigma_input)  # 计算用户选择参数下的正态分布PDF值

fig, ax = plt.subplots(figsize=(8, 5))  # 创建一个大小为8x5的绘图窗口

ax.plot(x_array, pdf_array, 'b', lw=1)  
# 绘制用户选择参数下的正态分布PDF曲线，蓝色线条，线宽为1

ax.axvline(x=mu_input, c='r', ls='--')  
# 在均值位置绘制一条红色虚线
ax.axvline(x=mu_input + sigma_input, c='r', ls='--')  
# 在均值加标准差位置绘制一条红色虚线
ax.axvline(x=mu_input - sigma_input, c='r', ls='--')  
# 在均值减标准差位置绘制一条红色虚线

# 绘制标准正态分布
ax.plot(x_array, uni_normal_pdf(x_array, 0, 1), c=[0.8, 0.8, 0.8], lw=1)  
# 绘制标准正态分布PDF曲线，灰色线条，线宽为1

ax.axvline(x=0, c=[0.8, 0.8, 0.8], ls='--')  
# 在标准正态分布均值位置绘制灰色虚线
ax.axvline(x=0 + 1, c=[0.8, 0.8, 0.8], ls='--')  
# 在标准正态分布均值加标准差位置绘制灰色虚线
ax.axvline(x=0 - 1, c=[0.8, 0.8, 0.8], ls='--')  
# 在标准正态分布均值减标准差位置绘制灰色虚线

ax.set_xlim(-5, 5)  # 设置x轴范围为[-5, 5]
ax.set_ylim(0, 1)  # 设置y轴范围为[0, 1]

ax.spines.right.set_visible(False)  # 隐藏右边框线
ax.spines.top.set_visible(False)  # 隐藏上边框线
ax.yaxis.set_ticks_position('left')  # 将y轴刻度设置在左侧
ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设置在底部
ax.tick_params(axis="x", direction='in')  # 设置x轴刻度线向内
ax.tick_params(axis="y", direction='in')  # 设置y轴刻度线向内

st.pyplot(fig)  # 在Streamlit应用程序中显示绘制的图表
