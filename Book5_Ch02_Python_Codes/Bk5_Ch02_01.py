
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.datasets import load_iris

# Load the iris data
iris_sns = sns.load_dataset("iris") 
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% Heatmap of X

plt.close('all')

# Visualize the heatmap of X

fig, ax = plt.subplots()
ax = sns.heatmap(X_df,
                 cmap='RdYlBu_r',
                 xticklabels=list(X_df.columns),
                 cbar_kws={"orientation": "vertical"},
                 vmin=-1, vmax=9)
plt.title('X')

#%% Histograms

fig, axes = plt.subplots(2,2)

sns.histplot(data=X_df, x = feature_names[0], binwidth = 0.2, ax = axes[0][0])
axes[0][0].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
sns.histplot(data=X_df, x = feature_names[1], binwidth = 0.2, ax = axes[0][1])
axes[0][1].set_xlim([0,8]); axes[0][1].set_ylim([0,40])
sns.histplot(data=X_df, x = feature_names[2], binwidth = 0.2, ax = axes[1][0])
axes[1][0].set_xlim([0,8]); axes[1][0].set_ylim([0,40])
sns.histplot(data=X_df, x = feature_names[3], binwidth = 0.2, ax = axes[1][1])
axes[1][1].set_xlim([0,8]); axes[1][1].set_ylim([0,40])

plt.tight_layout()

#%% draw multiple histograms on the same plot

fig, ax = plt.subplots()

sns.histplot(data=X_df, palette = "viridis", binwidth = 0.2)

fig, ax = plt.subplots()

sns.histplot(data=X_df, palette = "viridis",binwidth = 0.2,
             stat="density", common_norm=False)

#%% cumulative

fig, ax = plt.subplots()

sns.histplot(data=X_df, palette = "viridis",fill = False,
             binwidth = 0.2,element="step",
             cumulative=True, common_norm=False)

fig, ax = plt.subplots()

sns.histplot(data=X_df, palette = "viridis",fill = False,
             binwidth = 0.2,element="step",stat="density",
             cumulative=True, common_norm=False)

#%% variations of histograms

fig, ax = plt.subplots()

sns.histplot(data=X_df, palette = "viridis",fill = False,
             binwidth = 0.2,element="poly",stat="density", common_norm=False)

fig, ax = plt.subplots()

sns.histplot(data=X_df, palette = "viridis", binwidth = 0.2,
             element="step", kde = True,stat="density", common_norm=False)

#%% KDE

plt.tight_layout()

fig, ax = plt.subplots()
sns.kdeplot(data=X_df,fill=True, 
            common_norm=False, 
            alpha=.3, linewidth=1,
            palette = "viridis")


#%% bivariate

fig, ax = plt.subplots()
sns.histplot(iris_sns, x="sepal_length", y="sepal_width", bins = 20)

sns.displot(iris_sns, x="sepal_length", y="sepal_width", kind="kde", rug=True)

#%% variations of joint plots

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width",
              marginal_kws=dict(bins=20, fill=True))

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'hist', bins = 20,
              marginal_kws=dict(bins=20, fill=True))

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'hex', bins = 20,
              marginal_kws=dict(bins=20, fill=True))

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde')

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde', fill = True)

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'reg',
              marginal_kws=dict(bins=20, fill=True))

#%% multivariate pairwise

# without class labels
g = sns.pairplot(iris_sns)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_d") 
g.map_diag(sns.distplot, kde=False, color = 'b')

#%% Categorical data

#%% classes, univariate

for i in [0,1,2,3]:
    
    fig, ax = plt.subplots()
    sns.histplot(data=iris_sns, x=iris_sns.columns[i], hue="species", 
                 binwidth = 0.2, element="step")
    ax.set_xlim([0,8])

#%% classes, bivariate

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", hue="species")

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde', hue="species")

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde', fill = True, hue="species")

#%% Regression by classes

sns.lmplot(data = iris_sns, x="sepal_length", y="sepal_width", hue="species")


sns.lmplot(data = iris_sns, x="sepal_length", y="sepal_width", 
           hue="species", col="species")

#%% pairwise

# with class labels
g = sns.pairplot(iris_sns,hue="species", plot_kws={"s": 6}, palette = "viridis")
g.map_lower(sns.kdeplot)

#%% parallel coordinates

fig, ax = plt.subplots()
# Make the plot
pd.plotting.parallel_coordinates(iris_sns, 'species', colormap=plt.get_cmap("Set2"))
plt.show()

#%% Joy plot

import joypy
# you might have to install joypy

joypy.joyplot(iris_sns, ylim='own')


joypy.joyplot(iris_sns, column=['sepal_length', 'sepal_width',
              'petal_length', 'petal_width'], 
              by="species", ylim='own')


joypy.joyplot(iris_sns, by="species", column="sepal_width",
              hist=True, bins=40, overlap=0,grid=True)

#%% add mean values to the histograms

fig, axes = plt.subplots(2,2)

sns.histplot(data=X_df, x = feature_names[0], binwidth = 0.2, ax = axes[0][0])
axes[0][0].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
axes[0][0].vlines(x = X_df.mean()[feature_names[0]], 
                  ymin = 0, ymax = 40, color = 'r')

sns.histplot(data=X_df, x = feature_names[1], binwidth = 0.2, ax = axes[0][1])
axes[0][1].set_xlim([0,8]); axes[0][1].set_ylim([0,40])
axes[0][1].vlines(x = X_df.mean()[feature_names[1]], 
                  ymin = 0, ymax = 40, color = 'r')

sns.histplot(data=X_df, x = feature_names[2], binwidth = 0.2, ax = axes[1][0])
axes[1][0].set_xlim([0,8]); axes[1][0].set_ylim([0,40])
axes[1][0].vlines(x = X_df.mean()[feature_names[2]], 
                  ymin = 0, ymax = 40, color = 'r')

sns.histplot(data=X_df, x = feature_names[3], binwidth = 0.2, ax = axes[1][1])
axes[1][1].set_xlim([0,8]); axes[1][1].set_ylim([0,40])
axes[1][1].vlines(x = X_df.mean()[feature_names[3]], 
                  ymin = 0, ymax = 40, color = 'r')

plt.tight_layout()

#%% centroid added to jointplot

scatter_ax = sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width",
                           marginal_kws=dict(bins=20, fill=True))

scatter_ax.ax_joint.axvline(x=X_df.mean()[feature_names[0]], color = 'r')
scatter_ax.ax_joint.axhline(y=X_df.mean()[feature_names[1]], color = 'r')

scatter_ax.ax_joint.plot(X_df.mean()[feature_names[0]],
                        X_df.mean()[feature_names[1]],
                        marker = 'x', markersize = '12',
                        color = 'r')
scatter_ax.ax_joint.set_xlim(4,8)
scatter_ax.ax_joint.set_ylim(2,4.5)

#%% centroid added to jointplot, with classes

scatter_ax = sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", hue="species")

for label,color in zip(['setosa','versicolor','virginica'], ['b','r','g']):
    
    
    mu_x1_class = iris_sns.loc[iris_sns['species'] == label, 'sepal_length'].mean()
    mu_x2_class = iris_sns.loc[iris_sns['species'] == label, 'sepal_width'].mean()

    scatter_ax.ax_joint.axvline(x=mu_x1_class, color = color)
    scatter_ax.ax_joint.axhline(y=mu_x2_class, color = color)
    scatter_ax.ax_joint.plot(mu_x1_class, mu_x2_class,
                            marker = 'x', markersize = '12',
                            color = color)

#%% add mean values and std bands to the histograms

num = 0

fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
        
        mu  = X_df[feature_names[num]].mean()
        std = X_df[feature_names[num]].std()
        
        axes[i][j].axvline(x=mu, color = 'r')
        axes[i][j].axvline(x=mu - std, color = 'r')
        axes[i][j].axvline(x=mu + std, color = 'r')
        axes[i][j].axvline(x=mu - 2*std, color = 'r')
        axes[i][j].axvline(x=mu + 2*std, color = 'r')
        
        num = num + 1

#%% print the summary of iris data

print(iris_sns.describe(percentiles = [0.01, 0.25, 0.5, 0.75, 0.99]))

#%% 4-quantiles, quartiles

# visualize locations of three quartiles

num = 0

fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
        
        q75, q50, q25 = np.percentile(X_df[feature_names[num]], [75,50,25])
        axes[i][j].axvline(x=q75, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q25, color = 'r')
        
        num = num + 1

#%% 100-quantiles, percentile
# visualize two tails (1%, 99%)

num = 0
fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
        
        q1, q50, q99 = np.percentile(X_df[feature_names[num]], [1,50,99])
        axes[i][j].axvline(x=q1, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q99, color = 'r')
        
        num = num + 1

#%% box plot of data

fig, ax = plt.subplots()
sns.boxplot(data=X_df, palette="Set3")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

#%% violin plot of data

fig, ax = plt.subplots()

sns.violinplot(data=X_df, palette="Set3", bw=.2,
               cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

fig, ax = plt.subplots()

sns.swarmplot(data=X_df, palette="Set3", 
               linewidth=0.25, orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

#%% combine boxplot and swarmplot
fig, ax = plt.subplots()

sns.boxplot(data=X_df, orient="h")
sns.swarmplot(data=X_df, 
               linewidth=0.25, orient="h", color=".2")

#%% boxplot by labels

iris_long = iris_sns.melt(id_vars=['species'])
fig, ax = plt.subplots()
sns.boxplot(data=iris_long, x="value", y="variable", orient="h", 
            hue = 'species', palette="Set3")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

#%% Heatmap of covariance matrix

SIGMA = X_df.cov()

fig, axs = plt.subplots()

h = sns.heatmap(SIGMA,cmap='RdYlBu_r', linewidths=.05,annot=True)
h.set_aspect("equal")
h.set_title('Covariance matrix')

RHO = X_df.corr()

fig, axs = plt.subplots()

h = sns.heatmap(RHO,cmap='RdYlBu_r', linewidths=.05,annot=True)
h.set_aspect("equal")
h.set_title('Correlation matrix')

#%% skewness and kurtosis

print(X_df.skew())
print(X_df.kurt())

#%% compare covariance matrices

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

g1 = sns.heatmap(X_df[y==0].cov(),cmap="RdYlBu_r",
                 annot=True,cbar=False,ax=ax1,square=True,
                 vmax = 0.4, vmin = 0)
ax1.set_title('Y = 0, setosa')

g2 = sns.heatmap(X_df[y==1].cov(),cmap="RdYlBu_r",
                 annot=True,cbar=False,ax=ax2,square=True,
                 vmax = 0.4, vmin = 0)
ax2.set_title('Y = 1, versicolor')

g3 = sns.heatmap(X_df[y==2].cov(),cmap="RdYlBu_r",
                 annot=True,cbar=False,ax=ax3,square=True,
                 vmax = 0.4, vmin = 0)
ax3.set_title('Y = 2, virginica')

#%% compare correlation matrices

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

g1 = sns.heatmap(X_df[y==0].corr(),cmap="RdYlBu_r",
                 annot=True,cbar=False,ax=ax1,square=True,
                 vmax = 1, vmin = 0.15)
ax1.set_title('Y = 0, setosa')

g2 = sns.heatmap(X_df[y==1].corr(),cmap="RdYlBu_r",
                 annot=True,cbar=False,ax=ax2,square=True,
                 vmax = 1, vmin = 0.15)
ax2.set_title('Y = 1, versicolor')

g3 = sns.heatmap(X_df[y==2].corr(),cmap="RdYlBu_r",
                 annot=True,cbar=False,ax=ax3,square=True,
                 vmax = 1, vmin = 0.15)
ax3.set_title('Y = 2, virginica')
