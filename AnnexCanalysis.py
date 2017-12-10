# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:18:43 2014

@author: gerauddaspremont
"""
import numpy as np
import os
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
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


words = words = ['david cameron','weather','god','amazon','apple','energy','eu','war','google','election']

things = np.load('F1vsSparsity_LR.npz')
F1_mat = things['F1_mat']
tot_lab_mat = things['tot_lab_mat']
vec_sparsity = things['vec_sparsity']  
cs = [0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01] 
  
# à supprimer
#cs = list(np.linspace(1.0,0.1, num = 10, endpoint = True))
#F1_mat = np.random.random((len(words),len(cs)))
#tot_lab_mat = np.random.random((len(words),len(cs)))
#vec_sparsity = np.random.random((len(words),len(cs)))
#

plt.close('all')
i = 0
for w in words:
    Y = F1_mat[i,:]
    Y1 = vec_sparsity[i,:]
    Y2 = tot_lab_mat[i,:]
    X = cs
    
    
    fig = plt.figure(figsize=(23, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.15, top = 0.8, right = 0.82)
    
    lns = ax.plot(X, Y, color = 'b', linestyle = '-', linewidth = 3)
    
    title = r'$F_1$ versus sparsity for "'+ w.encode('string-escape') + r'"'
    y_labels = r'$F_1$ score'  
    ax.set_ylabel(y_labels , fontsize=30, labelpad = 80, rotation = 'horizontal')
    ax.set_xlabel(r'$\lambda$', fontsize = 30, labelpad = 10)
    #pdb.set_trace()
    
    #t = ax.set_title(title, fontsize=35)
    #t.set_y(1.05) 
    plt.text(0.5, 1.08, title,
             horizontalalignment='center',
             fontsize=35,
             transform = ax.transAxes)
             
    ax.set_xticks(X)   
    ax.tick_params(axis='x', labelsize = 20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.margins(0.1, 0)
    
    plt.xlim(1.02,0.08)
    
    #y_max = Y1.max() * 1.2
    plt.ylim((-0.1,1))
    
    ax2 = ax.twinx()
    lns1 = ax2.plot(X, Y1, color = 'b', linestyle = '--', linewidth = 3)
    lns2 = ax2.plot(X, Y2, color = 'g', linestyle = ':', linewidth = 3)
    ax2.set_ylabel("# of words", fontsize = 30, labelpad = 90, rotation = 'horizontal')
    plt.ylim(0,60)
    ax2.tick_params(axis='y', labelsize = 20)
    
    lnstot = lns + lns1 + lns2
    ax.legend(lnstot, (r'$F_1$ score','average # of words' ,'# of unique words'), 'upper left', prop={'size':20})
    
    
    path = '/Users/gerauddaspremont/Dropbox/project/thesis_1/Figures'
    filename = 'F1vslambda'+ w.replace(' ','')+'.pdf'
    filename = os.path.join(path, filename)       
    fig.savefig(filename, bbox_inches='tight')
    
    i+=1