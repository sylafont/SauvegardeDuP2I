# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:21:03 2024

@author: sybil
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

def plotGraph(root,titre,y, labels, color):
 
    fig, ax = plt.subplots()
    x = list(range(1,np.shape(y)[0]+1))
    ax.plot(x,y, label = labels)
    ax.set_title(titre)
    ax.legend()
    return IncorporateInTkWindow(root, fig, color)
    
    
def plotPie(root, titre, labels, sizes, color):
    colors = ["yellow", "blue", "green", "brown", "purple", "pink", "orange", "grey", "red", "cyan"]
    usedColors =[]
    
    for digit in labels:
        usedColors.append(colors[digit])
        
    proportionLabel = []
    for i, size in enumerate(sizes) : 
        proportion = round(size/sum(sizes)*100)
        proportionLabel.append(str(proportion) + "% "+str(labels[i]))
        
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    #ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    fig, ax = plt.subplots()
    wedges, text = ax.pie(sizes, wedgeprops=dict(width=0.5), colors = usedColors)
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        ax.annotate(proportionLabel[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)
        ax.set_title(titre)
    return IncorporateInTkWindow(root, fig, color)

def IncorporateInTkWindow(root, fig, color):
    fig.set_facecolor(color)
    canvas = FigureCanvasTkAgg(fig, root )
    canvas.draw()
    return fig, canvas.get_tk_widget()


    



