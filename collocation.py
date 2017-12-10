# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 23:09:54 2014

@author: gerauddaspremont
"""

import cPickle as pickle
import nltk
import sklearn
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown

regexp_tagger = nltk.RegexpTagger([(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),(r'(The|the|A|a|An|an)$', 'AT'),(r'.*able$', 'JJ'),(r'.*ness$', 'NN'),(r'.*ly$', 'RB'),(r'.*s$', 'NNS'),(r'.*ing$', 'VBG'),(r'.*ed$', 'VBD'),(r'.*', 'NN')])
ADJ = set(['JJ','OD','JJR','JJT','JJS','CD'])
NOUN = set(['NN','NNS','NP','NPS','NR','NRS'])
VRB = set(['VB','VBD','VBG','VBN','VBZ','DO','DOD','VBG','VBN','DOZ','HV','HVD','HVG','HVN','HVZ','BE','BED','BEDZ','BEG','BEN','BEZ','BEM','BER','MD'])
NVA = NVA.union(ADJ,NOUN,VRB)

def load_sparse_csc(filename):
    loader = np.load(filename)
    return sp.csc_matrix((  loader['data'], loader['indices'], loader['indptr'] ), shape = loader['shape'])

def tagged_corpus(corpus):
    brown_news_tagged = brown.tagged_sents(categories='news')
    tagger = nltk.UnigramTagger(brown_news_tagged, backoff = regexp_tagger)
    tag_corpus = tagger.tag(corpus)    
    filter_corpus = [w[0] for w in tag_corpus if w[1] in NVA]
    return filter_corpus
    
if False:
    corpus = pickle.load( open("list_of_words_not_in_dic.pkl","rb") )
else:
    raw = pickle.load( open("body.pkl","rb") )
    details = np.load('corpus_dictionaryTF.npz')
    dictionary = details['corpus_dictionary']
    dictionary = set(dictionary)
    words_NVA = pickle.load('list_of_NVA.pkl','rb')
    new_set = dictionary.intersection(NVA)
    tokenizer = RegexpTokenizer(r'\w+')
    corpus = [w for sublist in raw for w in tokenizer.tokenize(sublist.lower())]

print 'start collocations'
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(corpus)
del corpus
print 'filter finder'
#ignored_words = nltk.corpus.stopwords.words('english')
finder.apply_word_filter(lambda w: w not in new_set)
print 'start scoring'
scored = finder.score_ngrams(bigram_measures.student_t)

print 'filter results with t-test'

new_scored = [pairs for pairs in scored if pairs[1] > 2.576]
set_scored = set([pairs[0] for pairs in new_scored])

print 'saving...'
if True:
    with open('bigrams_scored2.pkl','wb') as f:
        pickle.dump(new_scored, f);
    with open('bigrams_no_scored2.pkl','wb') as f:
        pickle.dump(set_scored, f)
    
print 'done...'


