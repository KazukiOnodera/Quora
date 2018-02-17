# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:58:00 2017

@author: konodera
"""

import pandas as pd
import numpy as np
import utils

train, test = utils.load(1, 1)

#==============================================================================
# def
#==============================================================================
def set_diff(s1, s2):
    s1 = set(s1.lower().split())
    s2 = set(s2.lower().split())
    return s1-s2

def get_one_diff(s1, s2):
    s1 = set(s1.lower().split()) - set(['a','the'])
    s2 = set(s2.lower().split()) - set(['a','the'])
    if len(s1) == len(s2):
        if len(s1-s2)==len(s2-s1)==1:
            return ' -> '.join(sorted(list(s1-s2)+list(s2-s1)))
    return
    
def get_one_redundant(s1, s2):
    s1 = set(s1.lower().split()) - set(['a','the'])
    s2 = set(s2.lower().split()) - set(['a','the'])
    if len(s1)-len(s2)==1 and len(sorted(s1-s2))==1:
        return sorted(s1-s2)[0]
    elif len(s2)-len(s1)==1 and len(sorted(s2-s1))==1:
        return sorted(s2-s1)[0]
    return 



raise

df = train.sample(999)
df = train.ix[train.is_duplicate==1].sample(999)

df['one_diff'] = df.apply(lambda x: get_one_diff(x.q1, x.q2), axis=1)
df['one_redundant'] = df.apply(lambda x: get_one_redundant(x.q1, x.q2), axis=1)

df['q1_len'] = df.q1.str.count(' ')+1
df['q2_len'] = df.q2.str.count(' ')+1
df['q1_set_diff'] = df.apply(lambda x: set_diff(x.q1, x.q2), axis=1)
df['q2_set_diff'] = df.apply(lambda x: set_diff(x.q2, x.q1), axis=1)



train['one_diff'] = train.apply(lambda x: get_one_diff(x.q1, x.q2), axis=1)
train['one_redundant'] = train.apply(lambda x: get_one_redundant(x.q1, x.q2), axis=1)

diff = pd.crosstab(train.one_diff, train.is_duplicate)
redundant = pd.crosstab(train.one_redundant, train.is_duplicate)











# EDA

df = train[train.q1.str.contains('2016')]

#w = '(^{0}\s|\s{0}\s|^{0}$|\s{0}$)'.format('2016')
w = '(2014|2015|2016)'
w = 'mumbai'
train_ = train[train.q1.str.contains(w) | train.q2.str.contains(w)]
test_ = test[test.q1.str.contains(w) | test.q2.str.contains(w)]


w='my->thi'
train_ = train[train.one_diff == w]


idx = []
w1 = '2015'; w2='2016'
train_1 = train[train.q1.str.contains(w1) & ~train.q1.str.contains(w2) & train.q2.str.contains(w2) & ~train.q2.str.contains(w1)]
train_2 = train[train.q2.str.contains(w1) & ~train.q2.str.contains(w2) & train.q1.str.contains(w2) & ~train.q1.str.contains(w1)]
test_1 = test[test.q1.str.contains(w1) & ~test.q1.str.contains(w2) & test.q2.str.contains(w2) & ~test.q2.str.contains(w1)]
test_2 = test[test.q2.str.contains(w1) & ~test.q2.str.contains(w2) & test.q1.str.contains(w2) & ~test.q1.str.contains(w1)]


idx += test_1.test_id.tolist() + test_2.test_id.tolist()






print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

