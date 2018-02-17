# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:53:16 2017

@author: konodera
"""

import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

#import pandas as pd
#import numpy as np
#from itertools import product
#from collections import Counter
import multiprocessing as mp
total_proc = 2
import utils

#from gensim.models import KeyedVectors
#w2v = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)


train, test = utils.load(0, 1)

#==============================================================================
# def
#==============================================================================
def set_q(s):
    return set(set(s.lower().split()) - utils.kigou)

def _and_(s1, s2):
    return s1 & s2
    
def set_diff(s1, s2):
    return s1 - s2

def _or_(s1, s2):
    return s1 | s2

def main(p):
    """
    df = train.sample(9999)
    
    df.drop(['q1_set_diff','q2_set_diff'],axis=1,inplace=1)
    
    w = 'weight'
    df_ = df[df._and_.map(lambda x: w in x)]
    df_ = df[df.set_diff.map(lambda x: w in x)]
    
    
    w = 'any'
    df[df.set_diff.map(lambda x: w in x)].is_duplicate.mean()
    
    """
    if p==0:
        df = train
    elif p==1:
        df = test
    
    df['q1_'] = df.q1.map(set_q)
    df['q2_'] = df.q2.map(set_q)
    df['_and_'] = df.apply(lambda x: _and_(x.q1_, x.q2_), axis=1)
    df['q1_set_diff'] = df.apply(lambda x: set_diff(x.q1_, x.q2_), axis=1)
    df['q2_set_diff'] = df.apply(lambda x: set_diff(x.q2_, x.q1_), axis=1)
    df['set_diff'] = df.q1_set_diff.map(list) + df.q2_set_diff.map(list)
    df['_or_'] = df.apply(lambda x: _or_(x.q1_, x.q2_), axis=1)
    
    df.drop(['q1_','q2_'], axis=1, inplace=1)
    
    if 'is_duplicate' in df.columns:
        df.drop('is_duplicate', axis=1, inplace=1)
    
    path = '../input/mk/'
    name = 'and-or_0'
    if p==0:
        df.to_pickle(path+'train_{0}.p'.format(name))
        print('finish 000-8 train')
    elif p==1:
        df.to_pickle(path+'test_{0}.p'.format(name))
        print('finish 000-8 test')
    else:
        raise
    return
#==============================================================================
# main
#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))





print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

