

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import load_iris
from scipy.stats import norm
import scipy
import seaborn as sns


iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

x_array = np.linspace(0,8,100)
# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

MU = X_df.mean()
MU = np.array([MU]).T
SIGMA = X_df.cov()

# random data generator

from scipy.stats import multivariate_normal

multi_norm = multivariate_normal(MU[:,0], np.array(SIGMA))


num_rand = 200

X_rand = multi_norm.rvs(num_rand)

X_rand_df = pd.DataFrame(X_rand, columns = X_df.columns)


# without class labels
g = sns.pairplot(X_rand_df)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_d") 
g.map_diag(sns.distplot, kde=False, color = 'b')
