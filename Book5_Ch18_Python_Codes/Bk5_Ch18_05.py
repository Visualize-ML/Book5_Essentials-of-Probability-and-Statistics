# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 07:58:38 2022

@author: james
"""



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
import scipy.stats as st

#%% use seaborn to visualize the data

import seaborn as sns
# Load the iris data
iris_sns = sns.load_dataset("iris") 
# A copy from Seaborn

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width",hue = 'species',
              palette={'setosa': '#FF3300','versicolor': '#0099FF','virginica':'#8A8A8A'})


sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width",hue = 'species',
              kind="kde",
              palette={'setosa': '#FF3300','versicolor': '#0099FF','virginica':'#8A8A8A'})
