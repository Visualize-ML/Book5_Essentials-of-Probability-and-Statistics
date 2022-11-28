

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


from scipy.stats import multinomial
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 

num = 8
x1_array = np.arange(num + 1)
x2_array = np.arange(num + 1)

xx1, xx2 = np.meshgrid(x1_array, x2_array)

xx3 = num - xx1 - xx2
xx3 = np.where(xx3 >= 0.0, xx3, np.nan)

def heatmap_sum(data,i_array,j_array,title,vmin,vmax,cmap,annot = False):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    
    ax = sns.heatmap(data,cmap= cmap, #'YlGnBu', # YlGnBu
                     cbar_kws={"orientation": "horizontal"},
                     yticklabels=i_array, xticklabels=j_array,
                     ax = ax, annot = annot,
                     linewidths=0.25, linecolor='grey',
                     vmin = vmin, vmax = vmax,
                     fmt = '.3f')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.title(title)
    plt.yticks(rotation=0) 
    
    
#%% calculate multinomial probability

p_array = [0.6, 0.1, 0.3]
p_array = [0.3, 0.4, 0.3]
p_array = [0.1, 0.6, 0.3]

PMF_ff = multinomial.pmf(x=np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])).T, 
                         n=num, p=p_array)

PMF_ff = np.where(PMF_ff > 0.0, PMF_ff, np.nan)

PMF_ff = np.reshape(PMF_ff, xx1.shape)

#%% save to excel file

df = pd.DataFrame(np.flipud(PMF_ff))
filepath = 'PMF_ff.xlsx'
df.to_excel(filepath, index=False)

#%% 3D/2D scatter plot

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

ax.scatter3D(xx1.ravel(), xx2.ravel(), xx3.ravel(), 
             s = 400,
             marker='.',
             c=PMF_ff.ravel(), 
             cmap = 'RdYlBu_r')

# ax.contour(xx1, xx2, PMF_ff, 15, zdir='z', offset=0, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xticks([0,num])
ax.set_yticks([0,num])
ax.set_zticks([0,num])

ax.set_xlim(0, num)
ax.set_ylim(0, num)
ax.set_zlim3d(0, num)
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=-30, elev=20)
ax.view_init(azim=-90, elev=90)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))

ax.grid()
plt.show()

#%% heatmap

title = 'PMF of binomial distribution'
heatmap_sum(PMF_ff,x1_array,x2_array,title,0,0.12,'plasma_r',True)

#%% 3D stem chart

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')

ax.stem(xx1.ravel(), xx2.ravel(), PMF_ff.ravel(), basefmt=" ")

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('PMF')

ax.set_xlim((0,8))
ax.set_ylim((0,8))
ax.set_zlim((0,0.12))
# ax.set_zticks([])
# ax.grid(False)
ax.view_init(azim=-100, elev=20)
ax.set_proj_type('ortho')
plt.show()

# test only

# print(multinomial.pmf(x=(5,2,1), n=num, p=p_array))
