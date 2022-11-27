

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# utility functions

def plot_stem(df, column):
    
    pmf_df = df[column].value_counts().to_frame().sort_index()/len(df)
    
    fig, ax = plt.subplots()
    
    centroid = np.sum(pmf_df[column].astype(float) * pmf_df.index)
    plt.stem(pmf_df.index,pmf_df[column].astype(float))  
    plt.vlines(centroid, 0, 0.3, colors = 'r', linestyles = '--')
    plt.ylim(0, 0.3)
    plt.xlim(-5,40)
    plt.ylabel('Probability, PMF')
    plt.title(column + '; average = ' + '{0:.2f}'.format(centroid))

def plot_contour(XX1,XX2,df, column, XX1_fine, XX2_fine, YY):
    
    XX1_ = XX1.ravel()
    XX2_ = XX2.ravel()
    
    levels = np.sort(df[column].unique())
    print(list(levels)) # test only
    
    fig, ax = plt.subplots()
    plt.scatter(XX1_,XX2_)
    CS = plt.contour(XX1_fine,XX2_fine, 
                     YY,
                     levels = levels,
                     cmap = 'rainbow')
    
    plt.contour(XX1_fine,XX2_fine, 
                YY,
                levels = levels,
                cmap = 'rainbow')
    
    ax.clabel(CS, inline=True, 
              fontsize=12,
              fmt="%.2f")
    
    ax.set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xlim(1 - 0.5,6 + 0.5)
    plt.ylim(1 - 0.5,6 + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis="y", direction='in', length=8)
    ax.tick_params(axis="x", direction='in', length=8)
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.title(column)
    
#%% prepare data

X1_array = np.arange(1, 6 + 1)
X2_array = np.arange(1, 6 + 1)

X1_fine_array = np.linspace(0.5, 6.5, 100)
X2_fine_array = np.linspace(0.5, 6.5, 100)

XX1, XX2 = np.meshgrid(X1_array, X2_array)
XX1_fine, XX2_fine = np.meshgrid(X1_fine_array, X2_fine_array)

XX1_ = XX1.ravel()
XX2_ = XX2.ravel()


df = pd.DataFrame(np.column_stack((XX1_,XX2_)), columns = ['X1','X2'])


#%% X1 only

df['X1_sq'] = df['X1'] ** 2

YY_X1_only = XX1_fine

plot_contour(XX1,XX2, df, 'X1', XX1_fine, XX2_fine, YY_X1_only)

plot_stem(df, 'X1')

#%% X1 only squared

df['X1_sq'] = df['X1'] ** 2

YY_X1_sq = XX1_fine ** 2

plot_contour(XX1,XX2, df, 'X1_sq', XX1_fine, XX2_fine, YY_X1_sq)

plot_stem(df, 'X1_sq')

#%% sum: (X1 + X2)

df['sum'] = (df['X1'] + df['X2'])

YY_sum = XX1_fine + XX2_fine

plot_contour(XX1,XX2, df, 'sum', XX1_fine, XX2_fine, YY_sum)
plot_stem(df, 'sum')

#%% mean： (X1 + X2)/2

df['mean'] = (df['X1'] + df['X2'])/2

YY_mean = (XX1_fine + XX2_fine)/2

plot_contour(XX1,XX2, df, 'mean', XX1_fine, XX2_fine, YY_mean)
plot_stem(df, 'mean')

#%% mean： (X1 + X2 - 7)/2

df['mean_centered'] = (df['X1'] + df['X2'] - 7)/2

YY_mean_centered = (XX1_fine + XX2_fine - 7)/2

plot_contour(XX1,XX2, df, 'mean_centered', XX1_fine, XX2_fine, YY_mean_centered)
plot_stem(df, 'mean_centered')

#%% product of X1 and X2

df['product'] = df['X1'] * df['X2']

YY_product = XX1_fine * XX2_fine

plot_contour(XX1,XX2, df, 'product', XX1_fine, XX2_fine, YY_product)
plot_stem(df, 'product')

#%% devision, X1 over X2

df['devision'] = df['X1']/df['X2']

YY_devision = XX1_fine/XX2_fine

plot_contour(XX1,XX2, df, 'devision', XX1_fine, XX2_fine, YY_devision)
plot_stem(df, 'devision')

#%% difference

df['difference'] = df['X1'] - df['X2']

YY_difference = XX1_fine - XX2_fine

plot_contour(XX1,XX2, df, 'difference', XX1_fine, XX2_fine, YY_difference)

plot_stem(df, 'difference')

#%% abs_difference

df['abs_difference'] = np.abs(df['X1'] - df['X2'])

YY_abs_difference = np.abs(XX1_fine - XX2_fine)

plot_contour(XX1,XX2, df, 'abs_difference', XX1_fine, XX2_fine, YY_abs_difference)

plot_stem(df, 'abs_difference')

#%% (X1 - 3)**2 + (X2 - 3.5)**2

df['circle'] = (df['X1'] - 3.5) ** 2 + (df['X2'] - 3.5) ** 2

YY_circle = (XX1_fine - 3.5) ** 2 + (XX2_fine - 3.5) ** 2

plot_contour(XX1,XX2, df, 'circle', XX1_fine, XX2_fine, YY_circle)

plot_stem(df, 'circle')


