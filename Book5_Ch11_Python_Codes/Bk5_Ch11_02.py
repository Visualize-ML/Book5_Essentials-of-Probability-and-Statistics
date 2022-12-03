
###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

x = np.linspace(-4,4,num = 201)
y = np.linspace(-4,4,num = 201)
sigma_X = 1
sigma_Y = 2


xx,yy = np.meshgrid(x,y);

RHOs = np.linspace(-0.8,0.8,num = 9)

fig = plt.figure(figsize=(30,5))

for i in range(0,len(RHOs)):
    
    rho = RHOs[i]
    ax = fig.add_subplot(1,len(RHOs),int(i+1))
    ellipse = ((xx/sigma_X)**2 - 2*rho*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - rho**2);
    
    plt.contour(xx,yy,ellipse,levels = [1], colors = '#0099FF')

    A = (sigma_X**2 + sigma_Y**2)/2
    B = np.sqrt((rho*sigma_X*sigma_Y)**2 + ((sigma_X**2 - sigma_Y**2)/2)**2)
    length_major = np.sqrt(A + B)
    length_minor = np.sqrt(A - B)
    
    if sigma_X == sigma_Y and rho >= 0:
        theta = 45

    elif sigma_X == sigma_Y and rho < 0:
        theta = -45
    else:
        theta = 1/2*np.arctan(2*rho*sigma_X*sigma_Y/(sigma_X**2 - sigma_Y**2))
        theta = theta*180/np.pi
    
    if sigma_X >= sigma_Y:
        rect = Rectangle([-length_major, -length_minor] , 
                         width = 2*length_major, 
                         height = 2*length_minor,
                         edgecolor = 'k',facecolor="none",
                         transform=Affine2D().rotate_deg_around(*(0,0), theta)+ax.transData)
    else:
        rect = Rectangle([-length_minor, -length_major] , 
                         width = 2*length_minor, 
                         height = 2*length_major,
                         edgecolor = 'k',facecolor="none",
                         transform=Affine2D().rotate_deg_around(*(0,0), theta)+ax.transData)
    
    ax.add_patch(rect)

    X = np.linspace(-2.5,2.5,101)
    k = np.tan(theta/180*np.pi)
    axis_minor = k*X
    axis_major = - 1/k*X
    plt.plot(X,axis_minor, color = 'r', linewidth = 1.25)
    plt.plot(X,axis_major, color = 'r', linewidth = 1.25)


    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_position('zero')
    ax.spines['bottom'].set_color('none')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('\u03C1 = %0.1f' %rho)
