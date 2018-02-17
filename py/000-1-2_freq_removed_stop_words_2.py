# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 19:17:35 2017

@author: konodera
"""

import utils
import pandas as pd
import nltk
import gc
import multiprocessing as mp
total_proc = 4


train, test = utils.load(2, 1)

def mk_1q(train, test, col):
    
    train = train[col]
    test  = test[col]
    
    train = pd.melt(train)
    test  = pd.melt(test)
    
    df = pd.concat([train, test])
    
    df = df.value
    
    df.drop_duplicates(inplace=1)
    
    return df

def multi(p):
    li = []
    for i,s in enumerate(df):
        if i%total_proc != p:
            continue
        li += s.split()
    return li


df = mk_1q(train, test, ['q1', 'q2'])


pool = mp.Pool(total_proc)
callback = pool.map(multi, range(total_proc))
callback = sum(callback, [])

fd = nltk.FreqDist(callback)
del train, test, callback; gc.collect()

df = pd.DataFrame(fd.most_common(), columns=['word','cnt'])

df = df[df.word.str.len()>1]
df = df[df.cnt>1]
df = df[df.word.str.isalpha()]

df['weight'] = df.cnt.sum()/df.cnt
df['weight'] = (df['weight'] - df['weight'].mean()) / df['weight'].std()
df['weight'] = df['weight']-df['weight'].min()


df.to_csv('../nlp_source/tf.csv.gz', index=0,
          compression='gzip')


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))



