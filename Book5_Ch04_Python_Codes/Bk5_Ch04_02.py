
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
pd.options.mode.chained_assignment = None  # default='warn'

# Load the iris data
X_df = sns.load_dataset("iris")

#%% self-defined function

def heatmap_sum(data,i_array,j_array,title,vmin,vmax,cmap):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    
    ax = sns.heatmap(data,cmap= cmap, #'YlGnBu', # YlGnBu
                     cbar_kws={"orientation": "horizontal"},
                     yticklabels=i_array, xticklabels=j_array,
                     ax = ax, annot = True,
                     linewidths=0.25, linecolor='grey',
                     vmin = vmin, vmax = vmax)
    
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    
    ax.set_aspect("equal")
    plt.title(title)
    plt.yticks(rotation=0) 
    
#%% Prepare data

X_df.sepal_length = round(X_df.sepal_length*2)/2
X_df.sepal_width  = round(X_df.sepal_width*2)/2

print(X_df.sepal_length.unique())
sepal_length_array = X_df.sepal_length.unique()
sepal_length_array = np.sort(sepal_length_array)
print(X_df.sepal_length.nunique())

print(X_df.sepal_width.unique())
sepal_width_array = X_df.sepal_width.unique()
sepal_width_array = -np.sort(-sepal_width_array)
# sepal_width_array = np.flip(sepal_width_array)
print(X_df.sepal_width.nunique())

#%% scatter plot

fig, ax = plt.subplots()

# scatter plot of iris data
ax = sns.scatterplot(data = X_df, x = 'sepal_length', y = 'sepal_width')

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=0.5))
ax.set_yticks(np.arange(0, 6 + 1, step=0.5))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

#%% frequency count

X_df_12 = X_df[['sepal_length','sepal_width']]
X_df_12['count'] = 1
frequency_matrix = X_df_12.groupby(['sepal_length','sepal_width']).count().unstack(level=0)
frequency_matrix.columns = frequency_matrix.columns.droplevel(0)
frequency_matrix = frequency_matrix.fillna(0)
frequency_matrix = frequency_matrix.iloc[::-1]

probability_matrix = frequency_matrix/150
# frequency_matrix.to_excel('C:\\Users\\james\\Desktop\\' + 'frequency_matrix.xlsx')

#%% No species labels

title = 'No species label, frequency'
heatmap_sum(frequency_matrix,sepal_width_array,sepal_length_array,title,0,50,'plasma_r')

title = 'No species label, probability'
heatmap_sum(probability_matrix,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')

#%% Marginal, sepal length, X1

freq_sepal_length = frequency_matrix.sum(axis = 0).to_numpy().reshape((1,-1))
prob_sepal_length = probability_matrix.sum(axis = 0).to_numpy().reshape((1,-1))

title = 'Marginal count, frequency, sepal length'
heatmap_sum(freq_sepal_length,[],sepal_length_array,title,0,50,'plasma_r')

title = 'Marginal count, probability, sepal length'
heatmap_sum(prob_sepal_length,[],sepal_length_array,title,0,0.4,'viridis_r')

#%% Expectation of X1

E_X1 = prob_sepal_length @ sepal_length_array.reshape(-1,1)

E_X1_ = X_df['sepal_length'].mean() # test only

#%% Variance of X1

E_X1_sq = prob_sepal_length @ (sepal_length_array**2).reshape(-1,1)

var_X1 = E_X1_sq - E_X1**2

var_X1_ = X_df['sepal_length'].var()*149/150 # test only

#%% Marginal, sepal width, X2

freq_sepal_width = frequency_matrix.sum(axis = 1).to_numpy().reshape((-1,1))
prob_sepal_width = probability_matrix.sum(axis = 1).to_numpy().reshape((-1,1))

title = 'Marginal count, frequency, sepal width'
heatmap_sum(freq_sepal_width,sepal_width_array,[],title,0,50,'plasma_r')

title = 'Marginal count, probability, sepal width'
heatmap_sum(prob_sepal_width,sepal_width_array,[],title,0,0.4,'viridis_r')

#%% Expectation of X2

E_X2 = sepal_width_array.reshape(1,-1) @ prob_sepal_width

E_X2_ = X_df['sepal_width'].mean() # test only

#%% assumption: independence

title = 'Assumption: independence'
# joint probability
heatmap_sum(prob_sepal_width@prob_sepal_length,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')


#%% conditional probability, given sepal length

given_sepal_length = 5

prob_given_length = probability_matrix[given_sepal_length]
prob_given_length = prob_given_length/prob_given_length.sum()
prob_given_length = prob_given_length.to_frame()
title = 'No species label, probability given sepal length'
heatmap_sum(prob_given_length,sepal_width_array,[],title,0,0.4,'viridis_r')

#%% Matrix

probability_matrix_ = probability_matrix.to_numpy()

conditional_X2_given_X1_matrix = probability_matrix_/(np.ones((6,1))@np.array([probability_matrix_.sum(axis = 0)]))

title = 'X2 given X1'
heatmap_sum(conditional_X2_given_X1_matrix,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')

#%% conditional probability, given sepal width

given_sepal_width = 2.5

prob_given_width = probability_matrix.loc[given_sepal_width,:]
prob_given_width = prob_given_width/prob_given_width.sum()
prob_given_width = prob_given_width.to_frame().T
title = 'No species label, probability given sepal width'
heatmap_sum(prob_given_width,[],sepal_length_array,title,0,0.4,'viridis_r')


#%% Matrix

conditional_X1_given_X2_matrix = probability_matrix_/(probability_matrix_.sum(axis = 1).reshape(-1,1)@np.ones((1,8)))

title = 'X1 given X2'
heatmap_sum(conditional_X1_given_X2_matrix,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')


#%% Given Y

Given_Y = 'virginica' # 'setosa', 'versicolor', 'virginica'


fig, ax = plt.subplots()

# scatter plot of iris data
ax = sns.scatterplot(data = X_df.loc[X_df.species == Given_Y], x = 'sepal_length', y = 'sepal_width')

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=0.5))
ax.set_yticks(np.arange(0, 6 + 1, step=0.5))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

# joint probability of X1, X2 given Y
X_df_12_given_Y = X_df[['sepal_length','sepal_width','species']]
X_df_12_given_Y['count'] = 1

X_df_12_given_Y.loc[~(X_df_12_given_Y.species == Given_Y),'count'] = np.nan

X_df_12_given_Y = X_df_12_given_Y[['sepal_length','sepal_width','count']]
frequency_matrix_given_Y = X_df_12_given_Y.groupby(['sepal_length','sepal_width']).count().unstack(level=0)
frequency_matrix_given_Y.columns = frequency_matrix_given_Y.columns.droplevel(0)
frequency_matrix_given_Y = frequency_matrix_given_Y.fillna(0)
frequency_matrix_given_Y = frequency_matrix_given_Y.iloc[::-1]

probability_matrix_given_Y = frequency_matrix_given_Y/frequency_matrix_given_Y.sum().sum()

frequency_matrix_given_Y.to_excel('C:\\Users\\james\\Desktop\\' + 
                                  'frequency_matrix_given_' + 
                                  Given_Y + '.xlsx')

title = 'Given Y, frequency'
heatmap_sum(frequency_matrix_given_Y,sepal_width_array,sepal_length_array,title,0,50,'plasma_r')

title = 'Given Y, probability'
heatmap_sum(probability_matrix_given_Y,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')

# Conditional Marginal, sepal length

freq_sepal_length_given_Y = frequency_matrix_given_Y.sum(axis = 0).to_numpy().reshape((1,-1))
prob_sepal_length_given_Y = probability_matrix_given_Y.sum(axis = 0).to_numpy().reshape((1,-1))

title = 'Conditional Marginal count, frequency, sepal length'
heatmap_sum(freq_sepal_length_given_Y,[],sepal_length_array,title,0,50,'plasma_r')

title = 'Conditional Marginal count, probability, sepal length'
heatmap_sum(prob_sepal_length_given_Y,[],sepal_length_array,title,0,0.4,'viridis_r')

# Conditional Marginal, sepal width

freq_sepal_width_given_Y = frequency_matrix_given_Y.sum(axis = 1).to_numpy().reshape((-1,1))
prob_sepal_width_given_Y = probability_matrix_given_Y.sum(axis = 1).to_numpy().reshape((-1,1))

title = 'Conditional Marginal count, frequency, sepal width'
heatmap_sum(freq_sepal_width_given_Y,sepal_width_array,[],title,0,50,'plasma_r')

title = 'Conditional Marginal count, probability, sepal width'
heatmap_sum(prob_sepal_width_given_Y,sepal_width_array,[],title,0,0.4,'viridis_r')

# conditional independence

title = 'Assumption: conditional independence'
heatmap_sum(prob_sepal_width_given_Y@prob_sepal_length_given_Y,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')
