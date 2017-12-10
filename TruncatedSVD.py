# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:57:20 2014

@author: gerauddaspremont
"""

from sklearn.decomposition import TruncatedSVD
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pdb
from sklearn import cross_validation
import matplotlib.patches as mpatches
import os
import scipy.io as sio
from pylab import *
plt.close('all')
# functions to load data
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
def load_sparse_csc(filename):
    loader = np.load(filename)
    return sp.csc_matrix((  loader['data'], loader['indices'], loader['indptr'] ), shape = loader['shape'])

def load_sparse_csr(filename, tipe = np.float32):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr'] ), shape = loader['shape'], dtype=tipe)


n_features = 20
Sparseness = 'components' # 'data' or 'components'
things2 = np.load('corpus_dictionaryTFtopics.npz')
dictionary = things2['corpus_dictionary']
dates = things2['dates']
corpus = load_sparse_csr('corpus_TFIDFtopics.npz', tipe = np.float16)
if False:
    corpus = load_sparse_csr('corpus_TFIDFtopics.npz', tipe = np.float16)
    
    svd = TruncatedSVD(n_components=n_features, random_state=40, tol=1e-10)
    
    print 'start truncated svd computations'
    svdCorpus = svd.fit_transform(corpus)
    components = svd.components_
    print (np.count_nonzero(svd.components_ > 0) / (svd.components_.shape[0]*svd.components_.shape[1]))
    np.savez('TruncatedSVDmat', corpus = svdCorpus, components = svd.components_)
    # calculate a score:
    truc = svd.components_.sum(0)
    bob = svd.components_.max(0)
    scores = (truc - bob).mean(0)
else:
    things = np.load('TruncatedSVDmat.npz')
    components = things['components']
    svdCorpus = things['corpus']
F = []    
for i in range(components.shape[0]):
    #print
    top_scorer = np.argsort(-components[i,:])
    couples = [(dictionary[a], round(components[i,a], 3)) for a in top_scorer[:10]]
    bib = [dictionary[a] for a in top_scorer[:10]]
    F.append(bib)
    
F = []    
for i in range(components.shape[0]):
    #print
    top_scorer = np.argsort(-np.absolute(components[i,:]))
    couples = [(dictionary[a], round(components[i,a], 3)) for a in top_scorer[:10]]
    bib = [dictionary[a] for a in top_scorer[:10]]
    F.append(bib)


for i in range(len(F)):
    print '{} & ? & {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(i+1,F[i][0],F[i][1],F[i][2],F[i][3],F[i][4]
    ,F[i][5], F[i][6], F[i][7], F[i][8], F[i][9]), r'\\', '\n\\hline'

#for i in range(len(F)):
#    print '{} & {}, {}, {}, {}, {}, {}, {}, {}, {}, {},{} , {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(i+1,
#    F[i][0],F[i][1],F[i][2],F[i][3],F[i][4],F[i][5], F[i][6], F[i][7], F[i][8], F[i][9],
#    F[i][10],F[i][11],F[i][12],F[i][13],F[i][14],F[i][15], F[i][16], F[i][17], F[i][18], F[i][19]), r'\\', '\n\\hline'

topics_NMF = ['1', '2', '3', '4', '5','6','7','8',
              '9','10','11','12','13','14','15','16',
              'international','18','19','20']

#####################################
#     plot graph
####################################



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
# for multiple words start loop here:

#####################################
# setting classification parameters:
#####################################
#word = 'god'
#word = 'david cameron'
#word ='weather'
#word = 'amazon'
#word = 'apple'
#word = 'election' # Ou 'elections'
#word = 'love'
#word = 'eu'
#words = ['energy']
#word = 'war'
#word = 'google'
#word = 'god'

C_values = {}
C_values['cameron'] = 0
C_values['weather'] = 0
C_values['amazon'] = 0
C_values['apple'] = 0
C_values['election'] = 0
C_values['love'] = 0
C_values['eu'] = 0
C_values['energy'] = 0
C_values['war'] = 0
C_values['google'] = 0
C_values['god'] = 0


words = ['cameron','weather','god','amazon','apple','energy','eu','war','google','election']

c_vals = {}       
for word in words:
# doing logisitc regression for each period
    i = 0
    results = np.zeros((components.shape[0], len(periods)))
    name = word + '_topic_tracking'
    if False:
        zob = np.where(dictionary == word)
        labels = corpus[:,zob[0][0]].todense() != 0
        labels = np.asarray(labels)
        labels = labels.transpose()
        labels = labels[0]
        labels = labels.astype(np.int)
        for p in periods:
            selection = ( dates[0,:] == p[0] )
            selection2 = ( dates[1,:]==p[1])
            selection = np.logical_and(selection, selection2)
            #pdb.set_trace()
            error = np.zeros(2)
            c = 0.2
            test = True
            while (error.mean() < 0.1) or (test) :
                classifier = LogisticRegression(C=c, penalty='l1', tol=1e-10, class_weight = 'auto')
                model = classifier.fit(svdCorpus[selection,:], labels[selection])
                error = scores = cross_validation.cross_val_score(classifier, NMFCorpus[selection,:], labels[selection], cv = 2, scoring='recall')
                test  = model.coef_.ravel().sum() == 0
                c += 0.1
                if error.mean() == 0:
                    print word, c, p
            
            results[:,i] = model.coef_.ravel()
            np.save(name, results)
            C_values[word] = np.array([c, C_values[word]]).max() ;
            i+=1
    else:
        results = np.load(name+'.npy')
    ###########################################
    # start plot here
    ###########################################
    X = range(len(periods))
    Y = np.absolute(results)
    Y = np.clip(Y,0, Y.max())
    if Y.sum() == 0:
        pdb.set_trace()
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
    title = 'topic analysis for ' +'"' + word +'"' +  ' through time using LSA'  
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
    topics = topics[selection]
    topics = list(topics)
    if (selection.sum != len(subcolors)) & (len(subcolors) != len(p) ):
        pdb.set_trace()
    lgd = ax.legend(p , topics, loc='center', bbox_to_anchor=(1.15,0.5))
      
    path = '/Users/gerauddaspremont/Dropbox/project/thesis_1/Figures'
    filename = 'topictrackingSVD'+word+'.pdf'
    filename = os.path.join(path, filename)
    fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')