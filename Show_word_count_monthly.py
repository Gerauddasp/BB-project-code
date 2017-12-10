# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 19:11:42 2014

@author: gerauddaspremont
"""

import numpy as np
import os
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cross_validation
from sklearn import metrics
import numpy.matlib
from sklearn.decomposition import NMF
import cPickle as pickle
import itertools
import pdb
from sklearn.feature_selection import chi2
from matplotlib import rc
import re
plt.close('all')

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
def load_sparse_csc(filename):
    loader = np.load(filename)
    return sp.csc_matrix((  loader['data'], loader['indices'], loader['indptr'] ), shape = loader['shape'])

def load_sparse_csr(filename, tipe = np.float32):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr'] ), shape = loader['shape'], dtype=tipe)
    
details = np.load('corpus_dictionaryTF.npz')
dates = details['dates']
dictionary = np.load('big_dictionary.npy')

TF_mat = load_sparse_csr('corpus_TF_col.npz')

#words = ['david cameron','weather','barack obama','amazon','apple','microsoft','iraq','iran','afghanistan','russia']
words = ['david cameron','weather','god','amazon','apple','energy','eu','war','google','election']

years = np.unique(dates[0,:])
months = np.unique(dates[1,:])
half_year = []
half_year.append(months[:6])
half_year.append(months[6:])
quarters = []
quarters.append(months[:3])
quarters.append(months[3:6])
quarters.append(months[6:9])
quarters.append(months[9:])
x_labels = ['Q2-10','Q3-10','Q4-10','Q1-11','Q2-11','Q3-11','Q4-11',
            'Q1-12','Q2-12','Q3-12','Q4-12','Q1-13','Q2-13','Q3-13',
            'Q4-13','Q1-14']


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom', fontsize=15)

# setting things for latex
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


def plot_graph(x, w, ratio):
    #pdb.set_trace()
    N = len(x)
    ind = np.arange(N)
    width = .35
    
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(23, 10))
    ax = fig.add_subplot(111)
    rect = ax.bar(ind, x, width, color='b', align='center')
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif', serif=['Computer Modern Roman'])
    #pdb.set_trace()
    # add some
    title = '# of documents containing ' +'"' + w +'"' +  ' through time'  
    #title = re.escape(title)
    ax.set_ylabel('# of documents' ,color ='b', fontsize=30, labelpad = 5)
    #pdb.set_trace()
    ax.margins(0.04, 0)
    t = ax.set_title(title, fontsize=35)
    t.set_y(1.05) 
    ax.set_xticks(ind) 
    plt.xticks(np.arange(0,49,3))
    ax.set_xticklabels(x_labels, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    for tl in ax.get_yticklabels():
        tl.set_color('b')
    
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    y_max = np.array(count).max() * 1.2
    plt.ylim((0,y_max))
    
    ax2 = ax.twinx()
    ax2.plot(ind, ratio, 'ro-', linewidth=2, alpha = 0.3)
    ax.margins(0.04, 0)
    #ylab = r'$\frac{\# of documents containing ' + w.encode('string_escape') + r'}{number of document in period}$'
    ax2.set_ylabel('# of documents containing ' '"'+w+'"''/ \nnumber of document in period' , color='r', fontsize=30, labelpad = 10)
    #ax2.set_ylabel(ylab , color='r', fontsize=30, labelpad = 10)
    ax2.tick_params(axis='y', labelsize=20)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    
    #ax.legend( rect, '# of documents')
    
    #autolabel(rect, ax)
    #plt.show()
    return fig

# rajouter boucle ici
for w in words:
    plt.close('all')
    vec = TF_mat[:,np.where(dictionary == w)[0][0]].todense()
    count = []
    
    periods = [(y, q) for y in years for q in months]
    periods = periods[4:51]
    ratio = np.zeros(len(periods),dtype=np.float128)
    #pdb.set_trace()
    it = 0
    for p in periods: 
            selection = ( dates[0,:] == p[0] )
            selection2 = ( dates[1,:]==p[1])
            selection = np.logical_and(selection, selection2)
            num_docs = (vec[selection]>0)
            count.extend([num_docs.sum()])
            #pdb.set_trace()
            ratio[it]= float(num_docs.sum()) / selection.sum()
            it+=1
    
    fig = plot_graph(count, w, ratio)
    w = w.replace(' ','')
    path = '/Users/gerauddaspremont/Dropbox/project/thesis_1/Figures'
    filename = 'barplotmonthly'+w+'.pdf'
    filename = os.path.join(path, filename)       
    fig.savefig(filename, bbox_inches='tight')
    print w + ' done...'