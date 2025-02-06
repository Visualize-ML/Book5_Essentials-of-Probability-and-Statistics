
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import streamlit as st  # 导入Streamlit库，用于创建交互式Web应用程序
from scipy.stats import beta  # 从SciPy库中导入Beta分布相关函数
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import numpy as np  # 导入NumPy库，用于数值计算

x_array = np.linspace(0, 1, 200)  # 生成一个包含200个点的数组，范围在[0,1]，用于绘制Beta分布的PDF

with st.sidebar:  # 在Streamlit的侧边栏中添加内容
    st.write('Beta distribution PDF')  # 在侧边栏显示标题文字
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
             ''')  # 显示Beta分布概率密度函数的LaTeX公式
             
    alpha_input = st.slider('alpha', min_value=0.0, max_value=10.0, value=2.0, step=0.1)  # 创建一个滑块，用于调整参数α
    beta_input  = st.slider('beta', min_value=0.0, max_value=10.0, value=2.0, step=0.1)  # 创建另一个滑块，用于调整参数β

mean_loc  = alpha_input / (alpha_input + beta_input)  # 计算Beta分布的均值
pdf_array = beta.pdf(x_array, alpha_input, beta_input)  # 计算x_array对应的Beta分布PDF值

fig, ax = plt.subplots(figsize=(8, 8))  # 创建一个大小为8x8的图表

title_idx = '\u03B1 = ' + str(alpha_input) + '; \u03B2 = ' + str(beta_input)  # 设置标题，显示当前α和β的值
ax.plot(x_array, pdf_array, 'b', lw=1)  # 绘制Beta分布PDF曲线，蓝色线条，线宽为1

ax.axvline(x=mean_loc, c='r', ls='--')  # 在均值位置绘制一条红色虚线
ax.set_title(title_idx)  # 设置图表标题
ax.set_xlim(0, 1)  # 设置x轴范围为[0,1]
ax.set_ylim(0, 4)  # 设置y轴范围为[0,4]
ax.set_xticks([0, 0.5, 1])  # 设置x轴刻度为0, 0.5, 1
ax.set_yticks([0, 2, 4])  # 设置y轴刻度为0, 2, 4
ax.spines.right.set_visible(False)  # 隐藏右边框线
ax.spines.top.set_visible(False)  # 隐藏上边框线
ax.yaxis.set_ticks_position('left')  # 将y轴刻度设置在左侧
ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设置在底部
ax.tick_params(axis="x", direction='in')  # 设置x轴刻度向内
ax.tick_params(axis="y", direction='in')  # 设置y轴刻度向内

st.pyplot(fig)  # 在Streamlit应用中显示绘制的图表
