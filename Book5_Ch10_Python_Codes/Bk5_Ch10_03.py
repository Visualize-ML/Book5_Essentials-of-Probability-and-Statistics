

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

x = np.linspace(-4,4,num = 201)
y = np.linspace(-4,4,num = 201)
sigma_X = 1
sigma_Y = 2

xx,yy = np.meshgrid(x,y);

kk = np.linspace(-0.8,0.8,num = 9)

fig = plt.figure(figsize=(30,5))

for i in range(0,len(kk)):
    
    k = kk[i]
    ax = fig.add_subplot(1,len(kk),int(i+1))
    ellipse = ((xx/sigma_X)**2 - 2*k*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - k**2);
    
    plt.contour(xx,yy,ellipse,levels = [1], colors = '#0099FF')

    rect = Rectangle(xy = [- sigma_X, - sigma_Y] , 
                     width = 2*sigma_X, 
                     height = 2*sigma_Y,
                     edgecolor = 'k',facecolor="none")
    
    ax.add_patch(rect)
    

    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_position('zero')
    ax.spines['bottom'].set_color('none')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('\u03C1 = %0.1f' %k)
    
fig, ax = plt.subplots(figsize=(7, 7))

for i in range(0,len(kk)):
    
    k = kk[i]
    
    ellipse = ((xx/sigma_X)**2 - 2*k*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - k**2);
    
    plt.contour(xx,yy,ellipse,levels = [1], colors = '#0099FF')

rect = Rectangle(xy = [- sigma_X, - sigma_Y] , 
                 width = 2*sigma_X, 
                 height = 2*sigma_Y,
                 edgecolor = 'k',facecolor="none")
ax.add_patch(rect)

ax.set_xlim([-2.5,2.5])
ax.set_ylim([-2.5,2.5])
ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_position('zero')
ax.spines['bottom'].set_color('none')

