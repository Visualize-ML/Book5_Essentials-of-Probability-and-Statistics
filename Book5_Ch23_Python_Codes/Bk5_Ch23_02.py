

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
import scipy
import seaborn as sns

# Load the iris data
iris_sns = sns.load_dataset("iris") 
# A copy from Seaborn

#%% with no class labels

SIGMA = iris_sns.cov()
SIGMA = np.array(SIGMA)

MU    = iris_sns.mean()
MU    = np.array(MU)

g = sns.pairplot(iris_sns, diag_kind = 'kde',
                 kind='scatter', plot_kws={'alpha':0.5})
g.axes[0][0]

for i in [0, 1, 2, 3]:
    
    for j in [0, 1, 2, 3]:
        
        if i == j:
           pass
        else:


            ax = g.axes[i][j]
            mu_x = MU[j]
            mu_y = MU[i]
            
            ax.axvline(x = mu_x, color = 'r', linestyle = '--')
            ax.axhline(y = mu_y, color = 'r', linestyle = '--')
            ax.plot(mu_x,mu_y, color = 'k', marker = 'x', markersize = 12)
            
            sigma_X = np.sqrt(SIGMA[j][j])
            sigma_Y = np.sqrt(SIGMA[i][i])
            
            x = np.linspace(mu_x - 3.5*sigma_X,mu_x + 3.5*sigma_X,num = 201)
            y = np.linspace(mu_y - 3.5*sigma_Y,mu_y + 3.5*sigma_Y,num = 201)
            
            xx,yy = np.meshgrid(x,y);
            cov_X_Y = SIGMA[i][j]
            rho = cov_X_Y/sigma_X/sigma_Y
            
            ellipse = (((xx - mu_x)/sigma_X)**2 - 
                       2*rho*((xx - mu_x)/sigma_X)*((yy - mu_y)/sigma_Y) + 
                       ((yy - mu_y)/sigma_Y)**2)/(1 - rho**2);
            ellipse = np.sqrt(ellipse)
            
            ax.contour(xx,yy,ellipse,levels = [1,2,3], colors = 'r')


#%% with class labels

dimensions = ['sepal_length',
              'sepal_width',
              'petal_length',
              'petal_width']

g = sns.pairplot(iris_sns, hue="species",
                 kind='scatter', plot_kws={'alpha':0.5})
g.axes[0][0]

colors = ['b','r','g']

for i, i_dim in enumerate(dimensions):
    
    for j, j_dim in enumerate(dimensions):
        
        if i == j:
           pass
        else:

            ax = g.axes[i][j]
            
            for k, label in enumerate(iris_sns['species'].unique()):
                
                data = iris_sns.loc[iris_sns['species'] == label,
                                    [i_dim,j_dim]]
                
                mu_i_j = data.mean()
                mu_x = mu_i_j[1]
                mu_y = mu_i_j[0]
                
                SIGMA_i_j = data.cov()
                CORR_i_j  = data.corr()
                
                SIGMA_i_j = np.array(SIGMA_i_j)
                CORR_i_j  = np.array(CORR_i_j)
            
                ax.plot(mu_x,mu_y, color = colors[k], marker = 'x', markersize = 12)
                
                sigma_X = np.sqrt(SIGMA_i_j[1][1])
                sigma_Y = np.sqrt(SIGMA_i_j[0][0])
                
                x = np.linspace(mu_x - 3.5*sigma_X,mu_x + 3.5*sigma_X,num = 201)
                y = np.linspace(mu_y - 3.5*sigma_Y,mu_y + 3.5*sigma_Y,num = 201)
                
                xx,yy = np.meshgrid(x,y);
                rho = CORR_i_j[0][1]
                
                ellipse = (((xx - mu_x)/sigma_X)**2 - 
                           2*rho*((xx - mu_x)/sigma_X)*((yy - mu_y)/sigma_Y) + 
                           ((yy - mu_y)/sigma_Y)**2)/(1 - rho**2);
                
                ellipse = np.sqrt(ellipse)
                
                print(str(i_dim) + '_' + str(j_dim) + '_' + str(rho))
                
                ax.contour(xx,yy,ellipse,levels = [1,2,3], colors = colors[k])
