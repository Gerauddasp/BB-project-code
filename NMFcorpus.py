# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 11:45:37 2014

@author: gerauddaspremont
"""

from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse as sp

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


corpus = load_sparse_csr('corpus_TFIDFtopics.npz', tipe = np.float16)

nmf = NMF(n_components=n_features, tol=1e-04, sparseness = 'data', init='nndsvd', max_iter=200, nls_max_iter=2000)

print 'start NMF computations'
nmfCorpus = nmf.fit_transform(corpus)
print nmf.comp_sparseness_
print nmf.reconstruction_err_
print np.count_nonzero(nmf.components_ > 1)