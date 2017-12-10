# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 15:58:24 2014

@author: gerauddaspremont
"""


import numpy as np
import os
import scipy.sparse as sp
from sklearn import svm
import matplotlib.pylab as plt
import matplotlib
from sklearn import cross_validation
from sklearn import metrics
import numpy.matlib
from sklearn.decomposition import NMF
import cPickle as pickle
import itertools
import pdb
from sklearn.feature_selection import chi2


import matplotlib.patches as mpatches
frequency = [0.025, 0.0362, 0.0078, 0.0021, 0.0051, 0.0297, 0.031, 0.0463, 0.0062, 0.0346]

words = ['david cameron','weather','amazon','apple','energy','eu','war','google','election',
         'god']
##############################################         
# parameters for LR
#params = 
##############################################
# parameters for SVM
C_values = {}
C_values['david cameron'] = 0.2
C_values['weather'] = 0.19
C_values['amazon'] = 0.8
C_values['apple'] = 0.55
C_values['election'] = 0.15
C_values['love'] = 0.3
C_values['eu'] = 0.18
C_values['energy'] = 0.20
C_values['war'] = 0.13
C_values['google'] = 0.55
C_values['god'] = 0.6
params = [C_values[w] for w in words]
index = np.argsort(-np.array(params))
params = np.array(params)
frequency = np.array(frequency)
params = params[index]
frequency = frequency[index]








color=['b','r','g','y','c','m','chartreuse','burlywood','darkorange','deeppink']
       
x_labels = ['amazon','god', 'google', 'apple', 'david cameron \n energy', '', '', 'weather - eu', 'election', 'war' ]

fig = plt.figure(figsize=(23, 10))
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.15, top = 0.8, right = 0.82, bottom = 0.3)

lns = ax.scatter(params, frequency, color = color, s = 100, alpha = 1)

title = r'$\lambda$ versus $\frac{|\mathcal{D}_{t_q}|}{|\mathcal{D}|}$'
y_labels = r'$\frac{|\mathcal{D}_{t_q}|}{|\mathcal{D}|}$'
ax.set_ylabel(y_labels, fontsize = 30, labelpad = 50, rotation = 'horizontal')
ax.set_xlabel(r'$\lambda    $', fontsize = 30, labelpad = 5)
#pdb.set_trace()

#t = ax.set_title(title, fontsize=35)
#t.set_y(1.05) 
plt.text(0.5, 1.08, title,
         horizontalalignment='center',
         fontsize=35,
         transform = ax.transAxes)
 
 
ax.tick_params(axis='x', labelsize = 15)
plt.xticks(rotation=70)

xticks = ax.xaxis.get_major_ticks()
xticks[0].label1.set_visible(False)

ax.tick_params(axis='y', labelsize=15)


#ax.xaxis.set_label_coords(1.00, -0.03)
#ax.margins(0.1, 0)

plt.xlim(0,1)

#y_max = Y1.max() * 1.2
plt.ylim((-0.01,0.1))

p  = []
for i in color:
    p.extend([plt.Rectangle((0, 0), 1, 1, fc=i, alpha = 1)])

lgd = ax.legend(p , words, loc='center', bbox_to_anchor=(1.15,0.5))


path = '/Users/gerauddaspremont/Dropbox/project/thesis_1/Figures'
filename = 'lambdavsFreq2'+ w.replace(' ','')+'.pdf'
filename = os.path.join(path, filename)       
fig.savefig(filename, bbox_inches='tight')