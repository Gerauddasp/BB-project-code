# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:34:33 2014

@author: gerauddaspremont
"""

# import stuffs
import json
import os
import cPickle as pickle
import pdb
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()


def loop_through_files():
    All_body = []
    All_dates = []
    All_description = []
    All_titles = []
    indir = "/Users/gerauddaspremont/Dropbox/project/data/articles-415K"  
    for root, dirs, filenames in os.walk(indir, topdown = 'true'):
        folders =  os.path.relpath(root,indir)
        print folders
        for f in filenames:
            name = os.path.join(root, f)
            try:
                #pdb.set_trace()
                doc = open(name)
                doc = json.load(doc)
                All_body.append( doc["article"]["body"] )
                All_dates.append( doc["article"]["published"] )
                All_description.append( doc["article"]["description"] )
                All_titles.append( doc["article"]["title"] )
            except Exception:
                print "exception"
                pass
            if len(All_body)%1000 == 0:
                print len(All_body)
    return All_body, All_dates, All_description, All_titles

# what is going on: 
print "start running function"      
body, dates, description, titles  = loop_through_files()

print 'saving'
with open('body_lemma.pkl','wb') as f:
    pickle.dump(body, f);
    
with open('dates.pkl','wb') as f:
    pickle.dump(dates, f);
    
with open('description_lemma.pkl','wb') as f:
    pickle.dump(description, f);
    
with open('titles_lemma.pkl','wb') as f:
    pickle.dump(titles, f);