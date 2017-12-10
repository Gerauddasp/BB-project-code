# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 23:42:54 2014

@author: gerauddaspremont
"""
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pdb
from sklearn import cross_validation
import matplotlib.patches as mpatches
import os
import scipy.io as sio

plt.close('all')


topics_NMF = ['justice', 'video', 'crime', 'local politic', 'health','fire','police','education',
              'international','wales','floods','family','economy','syria','car accidents','movies',
              '17','18','scotland','national politics']
              
words = ['cameron','weather','god','amazon','apple','energy','eu','war','google','election']

things2 = np.load('corpus_dictionaryTFtopics.npz')
dictionary = things2['corpus_dictionary']
dates = things2['dates']
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
# ecluding the first period
x_labels = ['Q3-10','Q4-10','Q1-11','Q2-11','Q3-11','Q4-11',
            'Q1-12','Q2-12','Q3-12','Q4-12','Q1-13','Q2-13','Q3-13',
            'Q4-13','Q1-14']
periods = [(y, q) for y in years for q in months]
periods = periods[6:51]
mat_contents = sio.loadmat('smoothtopics.mat')

results_all = mat_contents['results']


for w in range(len(words)):
    results = results_all[:,:,w]
    ###########################################
    X = range(len(periods))
    Y = results.copy()
    Y = np.clip(Y,0, Y.max())
    #if Y.sum() == 0:
    #    pdb.set_trace()
    orders = np.argsort(Y.sum(1))
    Y = Y[orders[-3:],:]
    Y = np.divide(Y, np.matlib.repmat(Y.sum(0),3,1))
    Y = np.nan_to_num(Y)
    Y = np.cumsum(Y, axis = 0)
    selection = (Y.sum(1) != 0)
    orders = orders[-3:]
    ###########################################
    fig = plt.figure(figsize=(23, 10))
    fig.subplots_adjust(left=0.25, top = 0.9, right = 0.8)
    ax = fig.add_subplot(111)
    title = 'topic analysis for' +'"' + words[w] +'"' +  ' through time'  
    #title = re.escape(title)
    ax.set_ylabel('topics presence', fontsize=30, labelpad = 135, rotation = 'horizontal')
    #pdb.set_trace()
    ax.margins(0, 0)
    t = ax.set_title(title, fontsize=35)
    t.set_y(1.05)   
    plt.xticks(np.arange(0,46,3))
    ax.set_xticklabels(x_labels, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)    
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    
    color=['b','r','g','y','c','m','chartreuse','burlywood','chocolate','crimson', 'darkseagreen',
           'darkorange', 'deeppink', 'indigo', 'thistle','darkslategrey','darkviolet',
           'dodgerblue', 'tomato', 'silver']
    #cm = plt.get_cmap('hsv')
    #cm2 = plt.get_cmap('Set3')
    #NUM_COLORS = Y.shape[0];
    #ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    #color = []
    
    #for i in range(NUM_COLORS):
     #   if i%2==0:
        #    color.append(cm(1.*i/NUM_COLORS))
      #  else:
       #     color.append(cm2(1.*i/NUM_COLORS))
    
    ax.fill_between(X, 0, Y[0,:], facecolor=color[orders[0]], alpha = 1)
    for i in range(Y.shape[0]-1):
        ax.fill_between(X, Y[i,:], Y[i+1,:], facecolor=color[orders[i+1]], alpha = 1)
    
    p = []
    
    #pdb.set_trace()
    subcolors = np.array(color)
    #subcolors = subcolors[selection,:]
    subcolors = subcolors[orders]
    subcolors = list(subcolors)
    #subcolors = [tuple(el) for el in subcolors]
    for i in subcolors:
        p.extend([plt.Rectangle((0, 0), 1, 1, fc=i, alpha = 1)])
    
    #red_patch = mpatches.Patch(color='red', label='The red data')
    #plt.legend(handles=[red_patch], loc='center', bbox_to_anchor=(0.5,-0.1))
    topics = np.array(topics_NMF)
    topics = topics[orders]
    topics = list(topics)
    if (selection.sum != len(subcolors)) & (len(subcolors) != len(p) ):
        pdb.set_trace()
    lgd = ax.legend(p , topics, loc='center', bbox_to_anchor=(1.15,0.5))
      
    path = '/Users/gerauddaspremont/Dropbox/project/thesis_1/Figures'
    filename = 'topictrackingsmooth'+words[w]+'.pdf'
    filename = os.path.join(path, filename)
    fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')