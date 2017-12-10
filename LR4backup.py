# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:04:30 2014

@author: gerauddaspremont
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 23:23:48 2014

@author: gerauddaspremont
"""


# import stuffs
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

# word = 'david cameron'
#word = 'weather'
# word ='barack obama'
# word = 'amazon'
# word = 'apple'
# word = 'microsoft'
# word = 'iraq'
# word = 'iran'
word = 'afghanistan'


#words = ['david cameron','weather','barack obama','amazon','apple','microsoft','iraq','iran','afghanistan','russia']


# define model
#classifier = LogisticRegression(C=1, penalty='l1', tol=0.01)

# functions for model
def check_if(filename, word):
    test = False
    for n in filename:
        if n == word:
            test = True
    return test
 
def assemble_bigrams(coeffs, results):
    prov_coeffs = coeffs
    prov_results = results[:]
    test_list = list(itertools.permutations(results,2))
    new_labels = list(set(test_list).intersection(scored))
    to_delete = set()
    for pairs in new_labels:
        index1 = results.index(pairs[0])
        index2 = results.index(pairs[1])
        prov_results[index1] = pairs
        to_delete.add(index2)
        prov_coeffs[index1] = coeffs[[index1, index2]].max()
        
    np.delete(prov_coeffs, list(to_delete))
    prov_results = [w for w in prov_results if prov_results.index(w) not in to_delete]
    return prov_coeffs, prov_results

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
    #pdb.set_trace()
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
    classifier = LogisticRegression(C=0.2, penalty='l1', tol=1e-10, class_weight = 'auto')
    #pdb.set_trace()
    model = classifier.fit(new_corpus, labels)
    coeffs = model.coef_.ravel()
    indexes = np.argsort(coeffs)
    key_words = dic[indexes]
    results = key_words[np.nonzero(coeffs[indexes])].tolist()
    coeffs = coeffs[indexes]
    coeffs = coeffs[np.nonzero(coeffs)]
    #coeffs, results = assemble_bigrams(coeffs, results)
    print len(results)
    return coeffs, results
"""
def best_model(data, labels):
    # set max regularisation here
    regularisation = np.arange(0.1,0.5,0.1)
    folds = 5
    stratfield = cross_validation.StratifiedKFold(labels, n_folds=folds)
    results = np.zeros(len(regularisation) - 1)
    count = 0
    for c in regularisation[1:]:
        classifier = LogisticRegression(C=c, penalty='l1', tol=1e-10, class_weight = 'auto')
        scores = cross_validation.cross_val_score(classifier, data, labels, scoring='recall',
        cv=stratfield)
        results[count] = scores.mean()
        count += 1
    return regularisation[1:][results.argmax()]
"""
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
    #floor = np.percentile(matrix, 20)
    #floor = 0.2 * matrix.max()
    #nnz = np.nonzero(matrix)
    #matrix[nnz[0],nnz[1]] = np.clip(matrix[nnz[0],nnz[1]], floor, matrix.max())
    #############################
    matrix = matrix / matrix.max()
    
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(1,1,1)
    fig.subplots_adjust(left=0.25, top = 0.9, right = 0.6)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.binary, vmin = 0, aspect='auto')
    plt.colorbar()
    #pdb.set_trace()
    count = 0
    plt.yticks(range(len(y_labels)),y_labels)
    plt.xticks(range(len(x_labels)),x_labels)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5,10.5)
    mp_ticksize = matrix.sum(axis=1)
    count = 0
    for tick in ax.yaxis.get_major_ticks():
        pdb.set_trace()
        tick.label1.set_fontsize( ((mp_ticksize[count] / mp_ticksize.max()) * 10) + 10 )
        tick.label1.set
        count += 1
    plt.tight_layout()
    plt.show()

   
def runCode():
    words_mat = []
    coeff_mat = []
    for p in periods: 
        selection = ( dates[0,:] == p[0] )
        selection2 = np.asarray([False] * len(dates[1,:]))
        for subset in p[1]:
            selection2 = np.logical_or(selection2, (dates[1,:]==subset))
        selection = np.logical_and(selection, selection2)
        coeffs, features = extract_key_words( selection, 10, dictionary, corpus[selection,:])
        words_mat.append(features)
        coeff_mat.append(coeffs)

    return coeff_mat, words_mat
plt.close('all')
periods = [(y, list(q)) for y in years for q in quarters]
periods = periods[1:-3]
if False:
    coeff_mat, words_mat = runCode()
    y_labels, matrix = make_mat_for_graph(coeff_mat, words_mat)
    x_labels = ['Q2-10','Q3-10','Q4-10','Q1-11','Q2-11','Q3-11','Q4-11',
                'Q1-12','Q2-12','Q3-12','Q4-12','Q1-13','Q2-13','Q3-13',
                'Q4-13','Q1-14']
make_graph(matrix, y_labels, x_labels)