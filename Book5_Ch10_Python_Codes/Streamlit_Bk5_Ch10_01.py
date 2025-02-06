

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm # 导入 matplotlib 中的颜色映射功能
import streamlit as st # 导入 Streamlit 库，用于创建交互式 Web 应用程序

# 使用 Streamlit 创建侧边栏，允许用户调整参数
with st.sidebar:
    rho = st.slider('rho', min_value = -0.95, max_value = 0.95, value = 0.0, step = 0.05) # 相关系数 rho 的滑块
    sigma_X = st.slider('sigma_X', min_value = 0.5, max_value = 3.0, value = 1.0, step = 0.1) # X 的标准差滑块
    sigma_Y = st.slider('sigma_Y', min_value = 0.5, max_value = 3.0, value = 1.0, step = 0.1) # Y 的标准差滑块

# 设置均值 μ_X 和 μ_Y 为 0
mu_X = 0 
mu_Y = 0

# 显示标题和公式
st.write('Bivariate Gaussian distribution PDF') # 显示标题
st.latex(r'''{\displaystyle f(x,y)={\frac {1}{2\pi \sigma _{X}\sigma _{Y}{\sqrt {1-\rho ^{2}}}}}\exp \left(-{\frac {1}{2(1-\rho ^{2})}}\left[\left({\frac {x-\mu _{X}}{\sigma _{X}}}\right)^{2}-2\rho \left({\frac {x-\mu _{X}}{\sigma _{X}}}\right)\left({\frac {y-\mu _{Y}}{\sigma _{Y}}}\right)+\left({\frac {y-\mu _{Y}}{\sigma _{Y}}}\right)^{2}\right]\right)}''')
# 显示双变量正态分布概率密度函数的数学公式

# 构造均值和协方差矩阵
mu    = [mu_X, mu_Y] # 均值向量
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], # 协方差矩阵
        [sigma_X*sigma_Y*rho, sigma_Y**2]]

# 定义绘图范围和网格
width = 4 # 定义 X 和 Y 的绘图范围宽度
X = np.linspace(-width,width,321) # 在 [-width, width] 范围内生成 321 个 X 坐标点
Y = np.linspace(-width,width,321) # 在 [-width, width] 范围内生成 321 个 Y 坐标点

XX, YY = np.meshgrid(X, Y) # 生成网格

XXYY = np.dstack((XX, YY)) # 将 X 和 Y 网格堆叠为三维数组
bi_norm = multivariate_normal(mu, Sigma) # 定义双变量正态分布

#%% 计算联合 PDF 并进行可视化
f_X_Y_joint = bi_norm.pdf(XXYY) # 计算联合概率密度函数的值

# 3D 可视化
fig = plt.figure() # 创建新的图形对象
ax = fig.add_subplot(projection='3d') # 添加 3D 子图

ax.plot_wireframe(XX,YY, f_X_Y_joint,
                  rstride=10, cstride=10, # 设置网格线步长
                  color = [0.3,0.3,0.3], # 设置颜色
                  linewidth = 0.25) # 设置网格线宽度

ax.contour3D(XX,YY, f_X_Y_joint,15, # 添加 3D 等高线
             cmap = 'RdYlBu_r') # 使用红黄蓝颜色映射

# 设置轴标签
ax.set_xlabel('$x$') 
ax.set_ylabel('$y$') 
ax.set_zlabel('$f_{X,Y}(x,y)$') 

ax.set_proj_type('ortho') # 设置正交投影
# 修改轴网格线的样式
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

# 设置绘图范围
ax.set_xlim(-width, width) 
ax.set_ylim(-width, width) 
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max()) 

# 设置视角
ax.view_init(azim=-120, elev=30) # 设置方位角和仰角
plt.tight_layout() # 调整布局以避免重叠

st.pyplot(fig) # 在 Streamlit 中显示图形

#%% 绘制填充等高线图
fig, ax = plt.subplots(figsize=(7, 7)) # 创建新的图形和子图

# 绘制双变量正态分布的填充等高线
plt.contourf(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r) # 使用填充等高线
plt.axvline(x = mu_X, color = 'k', linestyle = '--') # 绘制垂直线表示 μ_X
plt.axhline(y = mu_Y, color = 'k', linestyle = '--') # 绘制水平线表示 μ_Y

# 设置轴标签
ax.set_xlabel('$x$') 
ax.set_ylabel('$y$') 

st.pyplot(fig) # 在 Streamlit 中显示图形

