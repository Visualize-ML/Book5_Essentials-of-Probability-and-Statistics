
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############


import streamlit as st # 导入 Streamlit，用于创建交互式 Web 应用程序
import matplotlib.pyplot as plt # 导入 Matplotlib，用于绘图
import numpy as np # 导入 NumPy，用于数值计算

# 在侧边栏中创建滑块，允许用户调整角度参数 theta
with st.sidebar:
    theta = st.slider('theta (degrees)', min_value=0, 
                      max_value=180, 
                      step=10, 
                      value=0) # theta 的滑块，单位为度
    theta = theta / 180 * np.pi # 将角度转换为弧度

# 定义圆上的点数量
num_points = 37
theta_array = np.linspace(0, 2*np.pi, num_points).reshape((-1, 1)) # 在 [0, 2π] 范围内生成等间隔角度数组

# 创建圆的点的坐标 (X, Y)
X_circle = np.column_stack((np.cos(theta_array), 
                            np.sin(theta_array))) # 圆上点的坐标
colors = plt.cm.rainbow(np.linspace(0, 1, len(X_circle))) # 为每个点分配颜色

# 定义更加密集的圆上的点
theta_array_fine = np.linspace(0, 2*np.pi, 500).reshape((-1, 1)) # 在 [0, 2π] 范围内生成更加密集的角度数组
X_circle_fine = np.column_stack((np.cos(theta_array_fine), 
                                 np.sin(theta_array_fine))) # 生成高分辨率的圆

# 定义一个单位方形
X_square = np.array([[0, 0],
                     [0, 1],
                     [1, 1],
                     [1, 0],
                     [0, 0]]) # 一个边长为 1 的单位方形

# 定义一个更大的中心在原点的方形
X_square_big = np.array([[1, 1],
                         [1, -1],
                         [-1, -1],
                         [-1, 1],
                         [1, 1]]) # 一个边长为 2 的大方形

center_array = X_circle * 0 # 将中心坐标数组初始化为全零

# 定义一个矩阵 A
A = np.array([[1.25, -0.75], 
              [-0.75, 1.25]]) # 用于生成协方差矩阵

SIGMA = A.T @ A # 计算协方差矩阵 SIGMA
L = np.linalg.cholesky(SIGMA) # 计算 SIGMA 的 Cholesky 分解
R = L.T # 获取上三角矩阵

# 创建绘图对象和轴
fig, ax = plt.subplots(figsize=(10, 10)) # 定义一个 10x10 尺寸的绘图

# 定义旋转矩阵 U，基于用户输入的 theta
U = np.array([[np.cos(theta), -np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]]) # 旋转矩阵

# 将方形应用旋转变换
X_square_R_rotated = X_square @ U # 单位方形旋转后的点
X_square_big_R_rotated = X_square_big @ U # 大方形旋转后的点

# 将圆点应用旋转和拉伸变换
X_R = X_circle @ U @ R # 应用旋转和线性变换的圆点
X_circle_fine_R = X_circle_fine @ U @ R # 高分辨率圆的旋转和线性变换
X_square_R = X_square @ U @ R # 单位方形的旋转和线性变换
X_square_big_R = X_square_big @ U @ R # 大方形的旋转和线性变换

# 绘制未变换的高分辨率圆
ax.plot(X_circle_fine[:, 0], X_circle_fine[:, 1], c=[0.8, 0.8, 0.8], lw=0.2) # 灰色细线表示原始圆

# 绘制原始圆上的点
ax.scatter(X_circle[:, 0], X_circle[:, 1], s=200, 
           marker='.', c=colors, zorder=1e3) # 用彩色点表示圆上的离散点

# 绘制未变换的小方形
ax.plot(X_square[:, 0], X_square[:, 1], c='k', linewidth=1) # 黑线表示单位方形
ax.plot(X_square_big[:, 0], X_square_big[:, 1], c='k') # 黑线表示大方形

# 绘制旋转变换后的方形
ax.plot(X_square_R[:, 0], X_square_R[:, 1], c='k', linewidth=1) # 旋转变换后的单位方形
ax.plot(X_square_big_R[:, 0], X_square_big_R[:, 1], c='k') # 旋转变换后的大方形

# 绘制旋转变换后的方形（仅旋转）
ax.plot(X_square_R_rotated[:, 0], X_square_R_rotated[:, 1], c='k', linewidth=1) # 单位方形旋转后
ax.plot(X_square_big_R_rotated[:, 0], X_square_big_R_rotated[:, 1], c='k') # 大方形旋转后

# 绘制变换后的高分辨率圆
ax.plot(X_circle_fine_R[:, 0], X_circle_fine_R[:, 1], c=[0.8, 0.8, 0.8], lw=0.2) # 灰色细线表示变换后的圆

# 绘制连接原始圆点和变换后圆点的线
ax.plot(([i for (i, j) in X_circle], [i for (i, j) in X_R]), 
        ([j for (i, j) in X_circle], [j for (i, j) in X_R]), 
        c=[0.8, 0.8, 0.8], lw=0.2) # 灰色线表示连接原始圆和变换后圆的点

# 绘制变换后的圆点
ax.scatter(X_R[:, 0], X_R[:, 1], s=200, 
           marker='.', c=colors, zorder=1e3) # 彩色点表示变换后的圆上的点

# 添加坐标轴
ax.axvline(x=0, c='k', lw=0.2) # 绘制垂直轴
ax.axhline(y=0, c='k', lw=0.2) # 绘制水平轴

# 设置坐标轴比例
ax.axis('scaled') # 确保坐标轴比例一致
ax.set_xbound(lower=-2.5, upper=2.5) # 设置 x 轴范围
ax.set_ybound(lower=-2.5, upper=2.5) # 设置 y 轴范围
ax.set_xticks([]) # 移除 x 轴刻度
ax.set_yticks([]) # 移除 y 轴刻度

# 隐藏图形的边框
ax.spines['top'].set_visible(False) # 隐藏顶部边框
ax.spines['right'].set_visible(False) # 隐藏右侧边框
ax.spines['bottom'].set_visible(False) # 隐藏底部边框
ax.spines['left'].set_visible(False) # 隐藏左侧边框

# 在 Streamlit 中显示绘图
st.pyplot(fig) # 渲染绘图到 Streamlit


