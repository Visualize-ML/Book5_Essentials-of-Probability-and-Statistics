

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
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
import scipy

iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

x_array = np.linspace(0,8,100)
# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% compare PDF curves

for feature in feature_names:
    
    sample = X_df[feature]
    
    hist = np.histogram(sample, bins=30, range = (0,8))
    hist_dist = scipy.stats.rv_histogram(hist)
    
    mu  = sample.mean()
    std = sample.std()
    
    N_pdf = norm.pdf(x_array, loc = mu, scale = std)

    epdf_y = hist_dist.pdf(x_array)

    fig, ax = plt.subplots()
    
    # plot empirical PDF
    plt.step(x_array,epdf_y, color = '#0070C0')
    ax.fill_between(x_array, epdf_y, step="pre", color = '#DBEEF3')
    
    plt.axvline(x=mu, color = 'r', linestyle = '--')
    plt.axvline(x=mu + std, color = 'r', linestyle = '--')
    plt.axvline(x=mu - std, color = 'r', linestyle = '--')
    
    # plot parametric (normal) PDF
    plt.plot(x_array,N_pdf, 'r')

    plt.xlabel(feature)
    plt.ylabel('PDF, probability density')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xlim(0,8)
    plt.ylim(0,1)
    plt.yticks(np.linspace(0,1,5))
    plt.grid(color = [0.7,0.7,0.7])



#%% compare CDF curves

for feature in feature_names:
    
    sample = X_df[feature]
    
    mu  = sample.mean()
    std = sample.std()
    
    N_cdf = norm.cdf(x_array, loc = mu, scale = std)

    ecdf = ECDF(sample)
    
    ecdf_y = ecdf(x_array)
    
    fig, ax = plt.subplots()
    
    # plot empirical CDF
    plt.step(x_array,ecdf_y)
    
    # plot parametric (normal) CDF
    plt.plot(x_array,N_cdf, 'r')
    
    plt.axvline(x=mu, color = 'r', linestyle = '--')
    plt.axvline(x=mu + std, color = 'r', linestyle = '--')
    plt.axvline(x=mu - std, color = 'r', linestyle = '--')
    
    plt.xlabel(feature)
    plt.ylabel('CDF, probability')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xlim(0,8)
    plt.ylim(0,1)
    plt.yticks(np.linspace(0,1,5))
    plt.grid(color = [0.7,0.7,0.7])
#%% convereted CDF scatter data, copula pairwise

import seaborn as sns 

# Load the iris data
iris_sns = sns.load_dataset("iris") 

iris_CDF_df = pd.DataFrame()

for index in np.arange(0,4):
    
    feature = iris_sns.columns[int(index)]
    sample = iris_sns[feature]
    
    hist = np.histogram(sample, bins = 1000, range = (0,8))
    hist_dist = scipy.stats.rv_histogram(hist)
    ecdf_y = hist_dist.cdf(sample)
    
    # option B
    # ecdf = ECDF(sample)
    # ecdf_y = ecdf(sample)

    iris_CDF_df[feature] = np.array(ecdf_y)

iris_CDF_df['species'] = iris_sns['species']
print(iris_CDF_df.head())


# fig, ax = plt.subplots()
g = sns.jointplot(data=iris_CDF_df, 
                  x = 'sepal_length', 
                  y = 'sepal_width',
                  xlim = (0,1), 
                  ylim = (0,1))

# with no class labels
g = sns.pairplot(iris_CDF_df)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_d") 
g.map_diag(sns.distplot, kde=False, color = 'b')

# g.axes[0,0].set_xlim((0,1))
# g.axes[0,1].set_xlim((0,1))
# g.axes[0,2].set_xlim((0,1))
# g.axes[0,3].set_xlim((0,1))

# g.axes[0,0].set_ylim((0,1))
# g.axes[1,0].set_ylim((0,1))
# g.axes[2,0].set_ylim((0,1))
# g.axes[3,0].set_ylim((0,1))


#%% compare ICDF curves

cdf_array = np.linspace(0.001,0.999,100)


for feature in feature_names:
    
    sample = X_df[feature]
    
    mu  = sample.mean()
    std = sample.std()
    
    x_icdf = norm.ppf(cdf_array, loc = mu, scale = std)

    ecdf = ECDF(sample)
    
    ecdf_y = ecdf(x_array)
    
    fig, ax = plt.subplots()
    
    # plot empirical ICDF
    plt.step(ecdf_y, x_array)
    
    # plot parametric (normal) ICDF
    plt.plot(cdf_array, x_icdf, 'r')
    
    plt.ylabel(feature)
    plt.xlabel('CDF, probability')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.ylim(0,8)
    plt.xlim(0,1)
    plt.xticks(np.linspace(0,1,5))
    plt.grid(color = [0.7,0.7,0.7])

#%% QQ plot

import pylab 
import scipy.stats as stats

for feature in feature_names:
    
    sample = X_df[feature]
    mu  = sample.mean()
    std = sample.std()
    
    fig = plt.figure(figsize=(7, 7))
    
    stats.probplot(sample, dist="norm", plot=pylab)
    pylab.show()
    
    plt.axhline(y=mu, color = 'r', linestyle = '--')
    plt.axhline(y=mu + std, color = 'r', linestyle = '--')
    plt.axhline(y=mu - std, color = 'r', linestyle = '--')
    plt.axhline(y=mu + 2*std, color = 'r', linestyle = '--')
    plt.axhline(y=mu - 2*std, color = 'r', linestyle = '--')
    
    plt.axvline(x=0, color = 'r', linestyle = '--')
    plt.axvline(x=1, color = 'r', linestyle = '--')
    plt.axvline(x=-1, color = 'r', linestyle = '--')
    plt.axvline(x=2, color = 'r', linestyle = '--')
    plt.axvline(x=-2, color = 'r', linestyle = '--')
    
    plt.xlabel('Theoretical (standard normal) quantiles')
    plt.ylabel('Empirical quantiles')


#%% generate Z score

z_array = np.linspace(-4,4,100)

Z_score_df = (X_df - X_df.mean()) /X_df.std()

for feature in feature_names:
    
    sample = Z_score_df[feature]
    
    hist = np.histogram(sample, bins=40, range = (-4,4))
    hist_dist = scipy.stats.rv_histogram(hist)
    
    mu  = sample.mean()
    std = sample.std()
    
    N_pdf = norm.pdf(z_array, loc = mu, scale = std)

    epdf_y = hist_dist.pdf(z_array)

    fig, ax = plt.subplots()
    
    # plot empirical PDF
    plt.step(z_array,epdf_y, color = '#0070C0')
    ax.fill_between(z_array, epdf_y, step="pre", color = '#DBEEF3')
    
    plt.axvline(x=mu, color = 'r', linestyle = '--')
    plt.axvline(x=mu + std, color = 'r', linestyle = '--')
    plt.axvline(x=mu - std, color = 'r', linestyle = '--')
    
    # plot parametric (normal) PDF
    plt.plot(z_array,N_pdf, 'r')

    plt.xlabel(feature.replace('X', 'Z') + ' ($\sigma$)')
    plt.ylabel('PDF, probability density')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xlim(-4,4)
    plt.ylim(0,1)
    plt.yticks(np.linspace(0,1,6))
    plt.grid(color = [0.7,0.7,0.7])


