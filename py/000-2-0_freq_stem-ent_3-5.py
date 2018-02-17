# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:12:16 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))


import pandas as pd
import gc
import multiprocessing as mp
total_proc = 8
import nltk
import utils
stops = utils.stops

#==============================================================================
print('word freq')
#==============================================================================
train, test = utils.load(3)

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

df.to_csv('../nlp_source/stem_freq.csv.gz', index=0,
          compression='gzip')

#==============================================================================
print('named_entities freq')
#==============================================================================
train, test = utils.load(5)

col = ['q1_named_entities','q2_named_entities']

train = train[col]
test  = test[col]

train = pd.melt(train)
test  = pd.melt(test)

df = pd.concat([train, test])

df = df[['value']]




def name_ent_tag(wt, tag):
    return [w for w,t in wt if t == tag]

df['named_ent_ORG'] = df.value.map(lambda x: name_ent_tag(x, 'ORG'))
df['named_ent_GPE'] = df.value.map(lambda x: name_ent_tag(x, 'GPE'))
df['named_ent_PERSON'] = df.value.map(lambda x: name_ent_tag(x, 'PERSON'))
df['named_ent_NORP'] = df.value.map(lambda x: name_ent_tag(x, 'NORP'))
df['named_ent_LOC'] = df.value.map(lambda x: name_ent_tag(x, 'LOC'))
df['named_ent_DATE'] = df.value.map(lambda x: name_ent_tag(x, 'DATE'))
df['named_ent_TIME'] = df.value.map(lambda x: name_ent_tag(x, 'TIME'))
df['named_ent_EVENT'] = df.value.map(lambda x: name_ent_tag(x, 'EVENT'))



def multi2(p):
    li = []
    for i,s in enumerate(df[target]):
        if i%total_proc != p:
            continue
        li += s
    return li

target = 'named_ent_ORG'
pool = mp.Pool(total_proc)
callback = pool.map(multi2, range(total_proc))
callback = sum(callback, [])

fd = nltk.FreqDist(callback)
dist = pd.DataFrame(fd.most_common(), columns=['word','cnt'])
dist = dist[dist.word.str.len()>1]

dist.to_csv('../nlp_source/ent_ORG_freq.csv.gz', index=0,
          compression='gzip')


target = 'named_ent_GPE'
pool = mp.Pool(total_proc)
callback = pool.map(multi2, range(total_proc))
callback = sum(callback, [])

fd = nltk.FreqDist(callback)
dist = pd.DataFrame(fd.most_common(), columns=['word','cnt'])
dist = dist[dist.word.str.len()>1]

dist.to_csv('../nlp_source/ent_GPE_freq.csv.gz', index=0,
          compression='gzip')



target = 'named_ent_PERSON'
pool = mp.Pool(total_proc)
callback = pool.map(multi2, range(total_proc))
callback = sum(callback, [])

fd = nltk.FreqDist(callback)
dist = pd.DataFrame(fd.most_common(), columns=['word','cnt'])
dist = dist[dist.word.str.len()>1]

dist.to_csv('../nlp_source/ent_PERSON_freq.csv.gz', index=0,
          compression='gzip')


target = 'named_ent_LOC'
pool = mp.Pool(total_proc)
callback = pool.map(multi2, range(total_proc))
callback = sum(callback, [])

fd = nltk.FreqDist(callback)
dist = pd.DataFrame(fd.most_common(), columns=['word','cnt'])
dist = dist[dist.word.str.len()>1]

dist.to_csv('../nlp_source/ent_LOC_freq.csv.gz', index=0,
          compression='gzip')


target = 'named_ent_EVENT'
pool = mp.Pool(total_proc)
callback = pool.map(multi2, range(total_proc))
callback = sum(callback, [])

fd = nltk.FreqDist(callback)
dist = pd.DataFrame(fd.most_common(), columns=['word','cnt'])
dist = dist[dist.word.str.len()>1]

dist.to_csv('../nlp_source/ent_EVENT_freq.csv.gz', index=0,
          compression='gzip')





print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))




















