# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 23:09:29 2017

@author: konodera
"""

import pandas as pd
import numpy as np
import utils
import gensim

train, test = utils.load(0,1)
train, test = utils.load(2,1)

df = train.sample(999)
model = gensim.models.KeyedVectors.load_word2vec_format('../nlp_source/GoogleNews-vectors-negative300.bin.gz', binary=True)


s1 = 'read find youtube comments'
s2 = 'see youtube comments'



s1 = 'How long do you boil crab legs'

for w1,w2 in zip(s1.split(),s1.split()[1:]):
    try:
        sim = model.similarity(w1,w2)
        print(w1, w2, sim)
    except:
        print(w1,w2,'pass')


w = 'account'
s = 'recover gmail password online'
for w_ in s.split():
    sim = model.similarity(w,w_)
    print(w, w_, sim)
    







print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

