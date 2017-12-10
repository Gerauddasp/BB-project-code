# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:38:09 2014

@author: gerauddaspremont
"""

import cPickle as pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer

print 'loading data...'

raw_dates = pickle.load( open("dates.pkl","rb") )
raw_corpus = pickle.load( open("body.pkl","rb") )
raw_titles = pickle.load( open('titles.pkl', "rb") )
raw_descriptions = pickle.load( open("description.pkl","rb") )

dates = []
months = []
for i in range( len(raw_dates) ):
    dates.append(raw_dates[i][0:4])
    months.append(raw_dates[i][5:7])
dates = map(str, dates)
months = map(str, months)
dates = map(int, dates)
months = map(int, months)

#check for empty elements
check1 = [i for i, x in enumerate(raw_corpus) if x == ""]
check2 = [i for i, x in enumerate(dates) if x == "[]"]

print check1
print check2

if check1 == check2:
    for p in check1:
        del raw_corpus[p]
        del dates[p]
        del months[p]
        del raw_titles[p]
        del raw_descriptions[p]
else:
    print "problem with lists"


dates = np.array(dates)
months = np.array(months)
date = np.vstack((dates, months))



def TF(text_list):
    #    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words = stopwords.words('english'))
    vectorizer = CountVectorizer(min_df=100, max_df=0.5 ,stop_words = stopwords.words('english'), dtype = np.int16)
    matrix = vectorizer.fit_transform(text_list)
    dictionnary = vectorizer.get_feature_names()
    return matrix, dictionnary

def TF_IDF(text_list, dictionnary):
    #vectorizer =  TfidfVectorizer(stop_words = stopwords.words('english'), vocabulary = dictionnary, dtype = np.float16)
    vectorizer = TfidfTransformer()
    matrix = vectorizer.fit_transform(text_list)
    return matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

# calling functions
print 'starting TF'
corpus_TF, corpus_dictionnary = TF(raw_corpus)
titles_TF, titles_dictionnary = TF(raw_titles)
descriptions_TF, descriptions_dictionnary = TF(raw_descriptions)

print 'saving...'

np.savez("corpus_dictionaryTF_svd", dates=date, corpus_dictionary = corpus_dictionnary)
np.savez("titles_dictionaryTFtopics_svd", dates=date, titles_dictionary = titles_dictionnary)
np.savez("descriptions_dictionaryTF_svd", dates=date, descriptions_dictionary = descriptions_dictionnary)


save_sparse_csr('corpus_TF_svd', corpus_TF)
save_sparse_csr('titles_TF_svd', titles_TF)
save_sparse_csr('descriptions_TF_svd', descriptions_TF)


print 'starting TF-IDF'
corpus_TFIDF = TF_IDF(raw_corpus, corpus_dictionnary)
titles_TFIDF = TF_IDF(raw_titles, titles_dictionnary)
descriptions_TFIDF = TF_IDF(raw_descriptions, descriptions_dictionnary)

print 'saving...'


save_sparse_csr('corpus_TFIDF_svd', corpus_TFIDF)
save_sparse_csr('titles_TFIDF_svd', titles_TFIDF)
save_sparse_csr('descriptions_TFIDF_svd', descriptions_TFIDF)