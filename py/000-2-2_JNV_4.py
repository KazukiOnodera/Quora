# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:34:25 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
import gc
import utils

train, test = utils.load(4, 1)

gc.collect()

#==============================================================================
# def
#==============================================================================

def J(wt):
    return [w for w,t in wt if t.startswith('J')]
    
def N(wt):
    return [w for w,t in wt if t.startswith('N')]
    
def V(wt):
    return [w for w,t in wt if t.startswith('V')]

def main(df):
    """
    df = train.sample(999)
    
    """
    
    df['q1_J'] = df.q1_wt.map(J)
    df['q2_J'] = df.q2_wt.map(J)
    df['q1_N'] = df.q1_wt.map(N)
    df['q2_N'] = df.q2_wt.map(N)
    df['q1_V'] = df.q1_wt.map(V)
    df['q2_V'] = df.q2_wt.map(V)
    
    return df.drop(['q1_wt', 'q2_wt'], axis=1)

#==============================================================================
train = main(train)
test  = main(test)



train.to_pickle('../input/mk/train_JNV.p')
test.to_pickle('../input/mk/test_JNV.p')


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

