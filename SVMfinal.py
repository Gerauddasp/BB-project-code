# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 19:40:08 2014

@author: gerauddaspremont
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 23:23:48 2014

@author: gerauddaspremont
"""

# pylint: disable=UndefinedMetricWarning


# import stuffs
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
 # global variables                        
details = np.load('corpus_dictionaryTF.npz')



corpus = load_sparse_csr('Corpus_TFIDF_col.npz')

# load bigrams score
scored = pickle.load( open("bigrams_no_scored2.pkl","rb") )

dates = details['dates']
#dictionary = details['corpus_dictionary']
dictionary = np.load('big_dictionary.npy')
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


#words = ['david cameron']
#words = ['weather']
#words = ['amazon']
#words = ['apple']
#words = ['election'] # Ou 'elections']
#words = ['love']
#words = ['eu']
#words = ['energy']
#words = ['war']
#words = ['google']
#words = ['god']


C_values = {}
C_values['david cameron'] = 0.05
C_values['weather'] = 0.04
C_values['amazon'] = 0.2
C_values['apple'] = 0.12
C_values['election'] = 0.035
C_values['love'] = 0.075
C_values['eu'] = 0.04
C_values['energy'] = 0.05
C_values['war'] = 0.03
C_values['google'] = 0.13
C_values['god'] = 0.135


words = ['david cameron','weather','god','amazon','apple','energy','eu','war','google','election',
         'god', 'love']


# define model
#classifier = LogisticRegression(C=1, penalty='l1', tol=0.01)

# functions for model
def check_if(filename, word):
    test = False
    for n in filename:
        if n == word:
            test = True
    return test
 

def remove_parts(dic, mat):
    index_to_delete = []
    if ' ' in set(word):
        unig = word.split(' ')
        to_delete = list([word])
        to_delete.extend(unig)
        for i in range(dictionary.shape[0]):
            test = [True for t in dic[i].split(' ') if t in set(to_delete)]
            #if dic[i] in set(to_delete):
            if any(test) == True:
                index_to_delete.extend([i])
    else:
        index_to_delete = np.where(dictionary == word)[0][0]
    mat[:,np.array(index_to_delete)] = 0
    #pdb.set_trace()
    return mat

def chi_square(mat, dic, labels):
    chi_values, p_values = chi2(mat, labels)
    test = (chi_values > 6.63)
    mat = mat[:,test]
    dic = dic[test]
    return mat, dic
   
def extract_key_words(selection, num_words, dic, new_corpus):
    #new_corpus = corpus[selection,:]
    zob = np.where(dic == word)
    labels = new_corpus[:,zob[0][0]].todense() != 0
    labels = np.asarray(labels)
    labels = labels.transpose()
    labels = labels[0]
    labels = labels.astype(np.int)
    
    new_corpus = remove_parts(dic, new_corpus)
    #new_corpus[:, zob[0][0]] = 0    
    #c = best_model(new_corpus, labels)
    #print c
    # NE PAS OUBLIER DE REMETTRE c A LA PLACE DU 0.2
    new_corpus, dic = chi_square(new_corpus, dic, labels)
    classifier = svm.LinearSVC(C=C_values[word], loss = 'l2', penalty = 'l1', dual=False, tol=1e-4, class_weight = 'auto')
    scores = cross_validation.cross_val_score(classifier, new_corpus, labels, scoring='f1',cv=3)
    scores = scores.mean()
    #pdb.set_trace()
    model = classifier.fit(new_corpus, labels)
    coeffs = model.coef_.ravel()
    indexes = np.argsort(coeffs)
    key_words = dic[indexes]
    results = key_words[np.nonzero(coeffs[indexes])].tolist()
    coeffs = coeffs[indexes]
    coeffs = coeffs[np.nonzero(coeffs)]
    #coeffs, results = assemble_bigrams(coeffs, results)
    ratio = float(labels.sum()) / float(selection.sum())
    return coeffs, results, len(results), scores, ratio

def make_mat_for_graph(coeffs, labels):
    seen = set()
    y_labels = []
    for lst in labels:
        sublist = [item for item in lst if item not in seen]
        seen.update(sublist)
        y_labels.append(sublist)
    y_labels = [item for sublist in y_labels for item in sublist if sublist is not []]
    #y_labels =list( set( [item for sublist in labels for item in sublist] ) )
    matrix = np.zeros(( len(y_labels), len(labels) ))
    column = -1
    for sub_list in labels:
        column += 1
        for words in sub_list:
            matrix[y_labels.index(words)][column] = coeffs[column][labels[column].index(words)]
    return y_labels, matrix
                

def make_graph(matrix, y_labels, x_labels):
    #############################
    # doing operation on matrix
    #floor = np.percentile(matrix, 0)
    floor = 0.05 * matrix.max()
    nnz = np.nonzero(matrix)
    matrix[nnz[0],nnz[1]] = np.clip(matrix[nnz[0],nnz[1]], floor, matrix.max())
    #############################
    #matrix = matrix / matrix.max()
    #pdb.set_trace()
    fig = plt.figure(figsize=(10,15))
    fig.subplots_adjust(left=0.2, top = 0.9, right = 0.8)
    ax = fig.add_subplot(1,1,1)   
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.binary, vmin = 0,
               aspect='auto')
    # 
    #plt.matshow(matrix, cmap=plt.cm.binary)
    #pdb.set_trace()
    plt.colorbar()
    plt.yticks(range(len(y_labels)),y_labels)
    plt.xticks(range(len(x_labels)),x_labels)
    #ax.set_xlim(0, len(x_labels))
    #ax.set_ylim(len(y_labels),0)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    title = r'$\ell_1$ penalty SVM for "' + word.encode('string_escape') + r'" - BBC news articles'
    t = ax.set_title(title, fontsize=15)
    t.set_y(1.05) 
    #####################################
    # set x ticker sizes
    #####################################
    mp_ticksize = matrix.sum(axis=1)
    count = 0
    plt.ylim( len(y_labels) +0.1, -1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize( ((mp_ticksize[count] / mp_ticksize.max()) * 10) + 10 )
        tick.label1.set_color(color = 'r')
        count += 1
    #####################################
    # plot lines between words and labels:
    #####################################
    if True:
        nonzeros = np.nonzero(matrix)
        nonzeros = np.matrix(nonzeros)
        nonzeros = np.hstack((nonzeros[0].T, nonzeros[1].T))
        x_coor = range(matrix.shape[0])
        y_coor = []
        for i in range(matrix.shape[0]):
            y_coor.extend([nonzeros[np.where(nonzeros[:,0]==i)[0][0],1].min()])
        ax.autoscale(False)
        for i in range(len(x_coor)):
            if y_coor[i] != 0:
                plt.plot( [-0.5,(y_coor[i] - 0.5)],[x_coor[i], x_coor[i]], linewidth = 1, alpha = 0.5, color = 'k', linestyle = ':')
    #plt.tight_layout()
    plt.show()
    return fig, ax

   
def runCode():
    words_mat = []
    coeff_mat = []
    average_len = []
    F1_scores = []
    ratios = []
    for p in periods: 
        selection = ( dates[0,:] == p[0] )
        selection2 = np.asarray([False] * len(dates[1,:]))
        for subset in p[1]:
            selection2 = np.logical_or(selection2, (dates[1,:]==subset))
        selection = np.logical_and(selection, selection2)
        selection = np.logical_and(selection, selection2)
        coeffs, features, length, s, rat = extract_key_words( selection, 10, dictionary, corpus[selection,:])
        words_mat.append(features)
        coeff_mat.append(coeffs)
        average_len.extend([length])
        F1_scores.extend([s])
        ratios.extend([rat])
    return coeff_mat, words_mat, np.array(average_len).mean(), np.array(s).mean(), np.array(rat).mean()
plt.close('all')
periods = [(y, list(q)) for y in years for q in quarters]
periods = periods[2:-3]
x_labels = ['Q3-10','Q4-10','Q1-11','Q2-11','Q3-11','Q4-11',
            'Q1-12','Q2-12','Q3-12','Q4-12','Q1-13','Q2-13','Q3-13',
            'Q4-13','Q1-14']
for word in words:
    if True:
        coeff_mat, words_mat, av_len, F1_s, r = runCode()
        y_labels, matrix = make_mat_for_graph(coeff_mat, words_mat)
        np.savez(word+'SVMres', y_labels = y_labels,matrix = matrix )
        print word + ' &',C_values[word], '& ', round(F1_s,2), '&' , round(av_len,2) , '& ', len(y_labels), '& ',  round(r, 4),  r' \\'
        print r'\hline'
    else:
        things = np.load(word+'SVMres.npz')
        y_labels = things['y_labels']
        matrix = things['matrix']
    fig, ax = make_graph(matrix, y_labels, x_labels)
    if True:
        word = word.replace(' ','')
        path = '/Users/gerauddaspremont/Dropbox/project/thesis_1/Figures'
        filename = 'BigGraphSVM'+word+'.pdf'
        filename = os.path.join(path, filename)       
        fig.savefig(filename, bbox_inches='tight')