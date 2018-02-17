# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:47:17 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
stops = set(list(utils.stops)+['one'])
import pandas as pd
import gc

train, test = utils.load(0)

train_, test_ = utils.load(4,1)

train = pd.merge(train, train_, on='id', how='left')
test = pd.merge(test, test_, on='test_id', how='left')

del train_, test_; gc.collect()

#==============================================================================
# def
#==============================================================================

def number(wt):
    ret = [w for w,t in wt if w.lower() not in stops and 'CD' in t]
    return list(set(ret))


def number_share(li, s):
    s = s.lower()
    return sum([1 for l in li if l.lower() in s])

def large(wt):
    ret = [w for w,t in wt[1:] if w.lower() not in stops and not w.islower() and w.isalpha()]
    return list(set(ret))

def large_share(li, s):
    s = s.lower()
    return sum([1 for l in li if l.lower() in s])
    
#def alphanum(wt):
#    return [w for w,t in wt if w.lower() not in stops and 'CD' in t and not w.isalpha() and not w.isdigit()]


def main(df):

    df['q1_number'] = df.q1_wt.map(number)
    df['q2_number'] = df.q2_wt.map(number)
    df['q1_number_len'] = df.q1_number.map(len)
    df['q2_number_len'] = df.q2_number.map(len)
    df['q1_num_share'] = df.apply(lambda x: number_share(x['q1_number'], x['q2']), axis=1)
    df['q2_num_share'] = df.apply(lambda x: number_share(x['q2_number'], x['q1']), axis=1)
    df['q1_num_share_ratio'] = df['q1_num_share']/df['q1_number_len']
    df['q2_num_share_ratio'] = df['q2_num_share']/df['q2_number_len']
    
    df['q1_large'] = df.q1_wt.map(large)
    df['q2_large'] = df.q2_wt.map(large)
    df['q1_large_len'] = df.q1_large.map(len)
    df['q2_large_len'] = df.q2_large.map(len)
    df['q1_large_share'] = df.apply(lambda x: number_share(x['q1_large'], x['q2']), axis=1)
    df['q2_large_share'] = df.apply(lambda x: number_share(x['q2_large'], x['q1']), axis=1)
    df['q1_large_share_ratio'] = df['q1_large_share']/df['q1_large_len']
    df['q2_large_share_ratio'] = df['q2_large_share']/df['q2_large_len']
    
    col = df.dtypes[df.dtypes!='object'].index.tolist()
    
    return df[col]

"""

df = train.sample(999)





"""
#==============================================================================
# main
#==============================================================================

train = main(train)
test = main(test)

utils.to_csv(train, test, 'f008')


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))


