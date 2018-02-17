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
from itertools import product
#from collections import Counter
import multiprocessing as mp
total_proc = 2
import utils

train, test = utils.load(3, 1)

#==============================================================================
# def
#==============================================================================
def set_q(s):
    return set(s.lower().split()) - utils.kigou

def _and_(s1, s2):
    return s1 & s2
    
def set_diff(s1, s2):
    return s1 - s2
    
def bigram(s1, s2):
    """
    s1 == q1_set_diff
    s2 == q2_set_diff
    
    return: product
    """
    
    return ['~'.join(sorted([w1,w2])) for w1,w2 in product(s1, s2)]


def _or_(s1, s2):
    return s1 | s2

def main(p):
    """
    df = train.sample(9999)
    
    df.drop(['q1_set_diff','q2_set_diff'],axis=1,inplace=1)
    
    w = 'weight'
    df_ = df[df._and_.map(lambda x: w in x)]
    df_ = df[df.set_diff.map(lambda x: w in x)]
    
    
    
    
    """
    if p==0:
        df = train
    elif p==1:
        df = test
    
    df['q1'] = df.q1.map(set_q)
    df['q2'] = df.q2.map(set_q)
    df['_and_'] = df.apply(lambda x: _and_(x.q1, x.q2), axis=1)
    df['q1_set_diff'] = df.apply(lambda x: set_diff(x.q1, x.q2), axis=1)
    df['q2_set_diff'] = df.apply(lambda x: set_diff(x.q2, x.q1), axis=1)
    df['set_diff'] = df.q1_set_diff.map(list) + df.q2_set_diff.map(list)
    df['set_diff_bigram'] = df.apply(lambda x: bigram(x.q1_set_diff, x.q2_set_diff), axis=1)
    df['_or_'] = df.apply(lambda x: _or_(x.q1, x.q2), axis=1)
    
    df.drop(['q1_set_diff','q2_set_diff'], axis=1, inplace=1)
    
    if 'is_duplicate' in df.columns:
        df.drop('is_duplicate', axis=1, inplace=1)
    
    path = '../input/mk/'
    name = 'and-or'
    if p==0:
        df.to_pickle(path+'train_{0}.p'.format(name))
        print('finish 000-7 train')
    elif p==1:
        df.to_pickle(path+'test_{0}.p'.format(name))
        print('finish 000-7 test')
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

