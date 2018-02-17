# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:43:55 2017

@author: konodera

nohup python -u 015.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
import numpy as np
import utils
from collections import Counter

n = 100

train, test = utils.load(3,1)

#==============================================================================
# def
#==============================================================================
def intersec(s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    return [w1 for w1 in s1 if w1 in s2]
    
def set_diff(s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    return [w1 for w1 in s1 if w1 not in s2]

def get_words():

    df = train.copy()
    df['intersec']= df.apply(lambda x: intersec(x.q1, x.q2), axis=1)
    df['q1_set_diff']= df.apply(lambda x: set_diff(x.q1, x.q2), axis=1)
    df['q2_set_diff']= df.apply(lambda x: set_diff(x.q2, x.q1), axis=1)
    
    
    df1 = df[df.is_duplicate==1]
    
    li = sum(df1.intersec.tolist(),[])
    cnt = Counter(li)
    
    df1_insec = pd.DataFrame(list(cnt.items()),
                             columns=['word', 'insec_cnt'])
    df1_insec['insec_rank'] = df1_insec.insec_cnt.rank()
    
    
    li = sum(df1.q1_set_diff.tolist(),[])
    li += sum(df1.q2_set_diff.tolist(),[])
    cnt = Counter(li)
    
    df1_setdiff = pd.DataFrame(list(cnt.items()),
                               columns=['word', 'setdiff_cnt'])
    df1_setdiff['setdiff_rank'] = df1_setdiff.setdiff_cnt.rank()
    
    
    df1_ = pd.merge(df1_insec, df1_setdiff, on='word',how='outer')
    df1_.fillna(0, inplace=1)
    
    df1_['imp'] = df1_.insec_rank - df1_.setdiff_rank
    
    
    df1_['cnt_sum'] = df1_.insec_cnt+df1_.setdiff_cnt
    df1_['insec_ratio'] = df1_.insec_cnt/df1_['cnt_sum']
    
#    df1_.sort_values('insec_ratio', ascending=0, inplace=1)
#    df1_ = df1_[df1_.insec_cnt>20]
    df1_.sort_values('setdiff_rank', ascending=0, inplace=1)
    
    df1_.reset_index(drop=1,inplace=1)
#    words = df1_.head(n).word.tolist()
    
#    return words
    return df1_

"""

def onlyone(q1,q2,w):
    q1 = q1.split()
    q2 = q2.split()
    if w in q1 and w in q2:
        return 0
    elif w in q1 or w in q2:
        return 1
    else:
        return 0

only one is '0'
ids = []
for w in words:
    w = '(^{0}\s|\s{0}\s|^{0}$|\s{0}$)'.format(w)
    ids += set(test[test.q1.str.contains(w) | test.q2.str.contains(w)].test_id.tolist())\
    -set(test[test.q1.str.contains(w) & test.q2.str.contains(w)].test_id.tolist())

set(ids) - set(tmp_te.test_id)

tmp_te = test[test.test_id.isin(ids)]



w = 'linkedin'
tmp_tr = train[train.q1.str.contains(w) | train.q2.str.contains(w)]
tmp_te = test[test.q1.str.contains(w) | test.q2.str.contains(w)]


w = '(^le\s|\sle\s|\sle$)'
tmp_tr = train[train.q1.str.contains(w) | train.q2.str.contains(w)]



"""
#==============================================================================
# 
#==============================================================================

#words = get_words(n)
#
#for w in words:
#    print(w)
#    train['bow_'+w] = train['q1'].map(lambda x: w in x.split())*1 + train['q2'].map(lambda x: w in x.split())*1
#    test['bow_'+w]  = test['q1'].map(lambda x: w in x.split())*1 + test['q2'].map(lambda x: w in x.split())*1
#
#utils.to_csv(train, test, 'f015')


df = get_words()
df.to_csv('../nlp_source/1-intersection.csv', index=0)


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

