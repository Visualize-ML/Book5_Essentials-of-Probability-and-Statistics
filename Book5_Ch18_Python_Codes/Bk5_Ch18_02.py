

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import load_iris

plt.close('all')

iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

axs = [ax1, ax2, ax3, ax4]

for name, ax in zip(feature_names, axs):
    
    df = X_df[name]
    KDE = sm.nonparametric.KDEUnivariate(df)
    KDE.fit(bw=0.5) # 0.1, 0.2, 0.4
    ax.fill_between(KDE.support, KDE.density, facecolor = '#DBEEF4')
    ax.plot(KDE.support, KDE.density)
    ax.scatter(df,0.03*np.abs(np.random.randn(df.size)),marker = 'x')

    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,8])
    ax.set_xlabel(name)


# Cumulative distribution
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

axs = [ax1, ax2, ax3, ax4]

for name, ax in zip(feature_names, axs):
    
    df = X_df[name]
    KDE = sm.nonparametric.KDEUnivariate(df)
    KDE.fit(bw=0.5) # 0.1, 0.2, 0.4
    ax.fill_between(KDE.support, KDE.cdf, facecolor = '#DBEEF4')
    ax.plot(KDE.support, KDE.cdf)
    ax.plot(KDE.support, KDE.density)

    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,8])
    ax.set_xlabel(name)
