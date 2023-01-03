

###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.datasets import load_iris

# Load the iris data
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

# KDE

fig, axes = plt.subplots(2,2)

sns.kdeplot(data=X_df,fill=True, x = feature_names[0], ax = axes[0][0])
axes[0][0].set_xlim([0,8]); axes[0][0].set_ylim([0,1])
sns.kdeplot(data=X_df,fill=True, x = feature_names[1], ax = axes[0][1])
axes[0][1].set_xlim([0,8]); axes[0][1].set_ylim([0,1])
sns.kdeplot(data=X_df,fill=True, x = feature_names[2], ax = axes[1][0])
axes[1][0].set_xlim([0,8]); axes[1][0].set_ylim([0,1])
sns.kdeplot(data=X_df,fill=True, x = feature_names[3], ax = axes[1][1])
axes[1][1].set_xlim([0,8]); axes[1][1].set_ylim([0,1])
