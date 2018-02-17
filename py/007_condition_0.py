# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:42:15 2017

@author: konodera

nohup python 008_condi.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
import pandas as pd
import multiprocessing as mp
total_proc = 2
from utils import find_noun_after_prep as find

from gensim.models import Doc2Vec
d2v = Doc2Vec.load('../nlp_source/d2v/enwiki_dbow/doc2vec.bin')
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)

train, test = utils.load(0,1)

#==============================================================================
# def
#==============================================================================
def condi(s1, s2, prep):
    """
    
    """
    # s1
    if prep not in s1.split():
        ret1 = []
    else:
        ret1 = find(prep, s1)
    # s2
    if prep not in s2.split():
        ret2 = []
    else:
        ret2 = find(prep, s2)
    
    return utils.list_match(ret1, ret2)

def condi_d2v(s1, s2, prep):
    """
    
    """
    try:
        # s1
        if prep not in s1.split():
            ret1 = []
        else:
            ret1 = find(prep, s1)
        # s2
        if prep not in s2.split():
            ret2 = []
        else:
            ret2 = find(prep, s2)
        
        return d2v.n_similarity(ret1, ret2)
    except:
        return -1


def condi_w2v(s1, s2, prep):
    """
    
    """
    try:
        # s1
        if prep not in s1.split():
            ret1 = []
        else:
            ret1 = find(prep, s1)
        # s2
        if prep not in s2.split():
            ret2 = []
        else:
            ret2 = find(prep, s2)
        
        return w2v.n_similarity(ret1, ret2)
    except:
        return -1


def main(df):
    """
    df = train.sample(999)
    
    """
    df['condi_in'] = df.apply(lambda x: condi(x.q1, x.q2, 'in'), axis=1)
    df['condi_for'] = df.apply(lambda x: condi(x.q1, x.q2, 'for'), axis=1)
    df['condi_from'] = df.apply(lambda x: condi(x.q1, x.q2, 'from'), axis=1)
    df['condi_under'] = df.apply(lambda x: condi(x.q1, x.q2, 'under'), axis=1)
    col = [c for c in df.columns if 'condi' in c]
    df = pd.get_dummies(df, columns=col)
    
    df['condi_d2v_in'] = df.apply(lambda x: condi_d2v(x.q1, x.q2, 'in'), axis=1)
    df['condi_d2v_for'] = df.apply(lambda x: condi_d2v(x.q1, x.q2, 'for'), axis=1)
    df['condi_d2v_from'] = df.apply(lambda x: condi_d2v(x.q1, x.q2, 'from'), axis=1)
    df['condi_d2v_under'] = df.apply(lambda x: condi_d2v(x.q1, x.q2, 'under'), axis=1)
    
    df['condi_w2v_in'] = df.apply(lambda x: condi_w2v(x.q1, x.q2, 'in'), axis=1)
    df['condi_w2v_for'] = df.apply(lambda x: condi_w2v(x.q1, x.q2, 'for'), axis=1)
    df['condi_w2v_from'] = df.apply(lambda x: condi_w2v(x.q1, x.q2, 'from'), axis=1)
    df['condi_w2v_under'] = df.apply(lambda x: condi_w2v(x.q1, x.q2, 'under'), axis=1)
    
    
    
    df.drop(['q1','q2'], axis=1, inplace=1)
    
    if 'is_duplicate' in df.columns:
        df.drop(['is_duplicate'], axis=1, inplace=1)
    
    if 'id' in df.columns:
        df.to_csv('~/Quora/feature/train_f007.csv.gz',
                  index=False, compression='gzip')
    else:
        df.to_csv('~/Quora/feature/test_f007.csv.gz',
                  index=False, compression='gzip')
        
    return

"""

train['condi_in'] = train.apply(lambda x: condi(x.q1, x.q2, 'in'), axis=1)
train['condi_for'] = train.apply(lambda x: condi(x.q1, x.q2, 'for'), axis=1)
train['condi_from'] = train.apply(lambda x: condi(x.q1, x.q2, 'from'), axis=1)

ct = pd.crosstab(train['condi_in'],train['is_duplicate'],normalize='index').sort_values(1,ascending=False)
ct.columns = ['p0','p1']
ct = pd.concat([ct,pd.crosstab(train['condi_in'],train['is_duplicate'])],axis=1)

"""
#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, [train, test])





print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))


