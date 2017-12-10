# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 17:31:31 2014

@author: gerauddaspremont
"""

import matplotlib.pyplot as plt
import numpy as np
import os
plt.close('all')
X = np.arange(-5, 6, 1)
Y1 = np.log(np.exp(1e-10*X) + np.exp(-1e-10*X))
Y2 = np.log(np.exp(1*X) + np.exp(-1*X))
Y3 = np.log(np.exp(10*X) + np.exp(-10*X))

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
fig.subplots_adjust(left=0.2, top = 0.9, right = 0.9)
t = ax0.set_title(r'Effect of $\rho$ parameter', fontsize=15)
t.set_y(1.05) 

ax0.plot(X, Y1)

ax1.plot(X, Y2)

ax2.plot(X, Y3)

ax0.tick_params(axis='y', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
ax0.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='x', labelsize=10)

ax0.set_ylabel(r'$\rho = 1e-10$', fontsize=30, labelpad = 1000 , rotation = 'horizontal')
ax1.set_ylabel(r'$\rho = 1$', fontsize=30, labelpad = 30, rotation = 'horizontal')
ax2.set_ylabel(r'$\rho = 10$' , fontsize=30, labelpad = 30, rotation = 'horizontal')


plt.text(-0.15, 0.5,r'$\rho = 1e-10$' ,
         horizontalalignment='center',
         fontsize=15,
         transform = ax0.transAxes)

plt.text(-0.15, 0.5,r'$\rho = 1$' ,
         horizontalalignment='center',
         fontsize=15,
         transform = ax1.transAxes)


plt.text(-0.15, 0.5,r'$\rho = 10$' ,
         horizontalalignment='center',
         fontsize=15,
         transform = ax2.transAxes)


# Hide the right and top spines
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax0.spines['left'].set_position('center')
ax1.spines['left'].set_position('center')
ax2.spines['left'].set_position('center')

# Only show ticks on the left and bottom spines
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

yticks = ax0.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks[-1].label1.set_visible(False)

yticks = ax1.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks[-1].label1.set_visible(False)

yticks = ax2.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks[-1].label1.set_visible(False)


# Tweak spacing between subplots to prevent labels from overlapping
plt.subplots_adjust(hspace=0.5)

path = '/Users/gerauddaspremont/Dropbox/project/thesis_1/Figures'
filename = 'rho.pdf'
filename = os.path.join(path, filename)       
fig.savefig(filename, bbox_inches='tight') #bbox_inches='tight'
