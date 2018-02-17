# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 02:37:54 2017

@author: konodera
"""

import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
import pandas as pd
from glob import glob
import multiprocessing as mp
total_proc = 10

#==============================================================================
# word
#==============================================================================
words = pd.read_csv('../nlp_source/stem_freq.csv.gz').head(3000).word.tolist()

train, test = utils.load(3)

def multi(p):
    train_ = train.copy()
    test_  = test.copy()
    ix = list(range(0,3000,300))[p]
    words_ = words[ix:ix+300]
    
    for w in words_:
        train_['BOW_'+w] = train_['q1'].map(lambda x: w in x.split())*1 + train_['q2'].map(lambda x: w in x.split())*1
        test_['BOW_'+w]  = test_['q1'].map(lambda x: w in x.split())*1 + test_['q2'].map(lambda x: w in x.split())*1

    utils.to_csv(train_, test_, 'f009-word-{0}'.format(p))



pool = mp.Pool(total_proc)
callback = pool.map(multi, range(total_proc))

#==============================================================================
# ents
#==============================================================================

train, test = utils.load(3)

files = sorted(glob('../nlp_source/ent*'))

for f in files:
    words = pd.read_csv(f).head(30).word.tolist()
    for w in words:
        train['ent_'+w] = train['q1'].map(lambda x: w.lower() in x.lower().split())*1 + train['q2'].map(lambda x: w.lower() in x.lower().split())*1
        test['ent_'+w]  = test['q1'].map(lambda x: w.lower() in x.lower().split())*1 + test['q2'].map(lambda x: w.lower() in x.lower().split())*1

utils.to_csv(train, test, 'f009-ent')


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

