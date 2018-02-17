# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:03:47 2017

@author: konodera
"""

import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
stops = utils.stops
import pandas as pd
import gc
import multiprocessing as mp
total_proc = 2

train, test = utils.load(4,1)

train_, test_ = utils.load(0)
train = pd.merge(train, train_, on='id', how='left')
test  = pd.merge(test, test_, on='test_id', how='left')

train_, test_ = utils.load(5)
train = pd.merge(train, train_, on='id', how='left')
test  = pd.merge(test, test_, on='test_id', how='left')

train_, test_ = utils.load(6)
train = pd.merge(train, train_, on='id', how='left')
test  = pd.merge(test, test_, on='test_id', how='left')

del train_, test_; gc.collect()

train.dropna(inplace=True)
test.dropna(inplace=True)

#==============================================================================
# def
#==============================================================================

def head_5W1H(wt):
    w,t = wt[0]
    if w in ['When','Where','Why','How','What','Who','Is','If','Are']:
        return w
    return

def first_V(wt):
    for w,t in wt:
        if w.lower() in stops:
            continue
        if t.startswith('V'):
            return w
    return
    

def count(wt, tag):
    return sum([1 for w,t in wt if t.startswith(tag) and w not in stops])

def count_prep(wt, prep):
    return sum([1 for w,t in wt if w == prep])

def name_ent_share(wt, s):
    return sum([1 for w,t in wt if w.lower() in s])

def name_ent_tag(wt, tag):
    return [w for w,t in wt if t == tag]
    
def name_ent_tag_share(ents, s):
    s = s.lower()
    return sum([1 for w in ents if w.lower() in s])

def noun_chunks_share(ch, s):
    s = s.lower()
    return sum([1 for w in ch if w.lower() in s])

def in_place(s, gpe):
    places = []
    s = s.lower()
    gpe = list(map(lambda x: x.lower(),gpe))
    if 'in' in s:
        places = utils.find_noun_after_prep('in', s)
        places = [p for p in places if p in gpe]
    return places



def main(p):
    
    if p==0:
        df = train
    elif p==1:
        df = test

    df['q1_head_5W1H'] = df.q1_wt.map(head_5W1H)
    df['q2_head_5W1H'] = df.q2_wt.map(head_5W1H)
    df = pd.get_dummies(df, columns=['q1_head_5W1H', 'q2_head_5W1H'])
    
    
    df['q1_count_V'] = df.q1_wt.map(lambda x: count(x,'V'))
    df['q2_count_V'] = df.q2_wt.map(lambda x: count(x,'V'))
    df['q1_count_N'] = df.q1_wt.map(lambda x: count(x,'N'))
    df['q2_count_N'] = df.q2_wt.map(lambda x: count(x,'N'))
    df['q1_count_J'] = df.q1_wt.map(lambda x: count(x,'J'))
    df['q2_count_J'] = df.q2_wt.map(lambda x: count(x,'J'))
    
    df['q1_count_in'] = df.q1_wt.map(lambda x: count_prep(x,'in'))
    df['q2_count_in'] = df.q2_wt.map(lambda x: count_prep(x,'in'))
    df['q1_count_on'] = df.q1_wt.map(lambda x: count_prep(x,'on'))
    df['q2_count_on'] = df.q2_wt.map(lambda x: count_prep(x,'on'))
    df['q1_count_for'] = df.q1_wt.map(lambda x: count_prep(x,'for'))
    df['q2_count_for'] = df.q2_wt.map(lambda x: count_prep(x,'for'))
    df['q1_count_at'] = df.q1_wt.map(lambda x: count_prep(x,'at'))
    df['q2_count_at'] = df.q2_wt.map(lambda x: count_prep(x,'at'))
    df['q1_count_under'] = df.q1_wt.map(lambda x: count_prep(x,'under'))
    df['q2_count_under'] = df.q2_wt.map(lambda x: count_prep(x,'under'))
    df['q1_count_and'] = df.q1_wt.map(lambda x: count_prep(x,'and'))
    df['q2_count_and'] = df.q2_wt.map(lambda x: count_prep(x,'and'))
    
    # named entities
    df['q1_named_ent_len'] = df.q1_named_entities.map(lambda x:len(x))
    df['q2_named_ent_len'] = df.q2_named_entities.map(lambda x:len(x))
    df['q1_named_ent_share'] = df.apply(lambda x:(name_ent_share(x.q1_named_entities, x.q2)), axis=1)
    df['q2_named_ent_share'] = df.apply(lambda x:(name_ent_share(x.q2_named_entities, x.q1)), axis=1)
    df['q1_named_ent_ratio'] = df.q1_named_ent_share / df.q1_named_ent_len
    df['q2_named_ent_ratio'] = df.q2_named_ent_share / df.q2_named_ent_len
    
    df['q1_named_ent_ORG'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'ORG'))
    df['q2_named_ent_ORG'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'ORG'))
    df['q1_named_ent_GPE'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'GPE'))
    df['q2_named_ent_GPE'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'GPE'))
    df['q1_named_ent_PERSON'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'PERSON'))
    df['q2_named_ent_PERSON'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'PERSON'))
    df['q1_named_ent_NORP'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'NORP'))
    df['q2_named_ent_NORP'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'NORP'))
    df['q1_named_ent_LOC'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'LOC'))
    df['q2_named_ent_LOC'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'LOC'))
    df['q1_named_ent_DATE'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'DATE'))
    df['q2_named_ent_DATE'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'DATE'))
    df['q1_named_ent_TIME'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'TIME'))
    df['q2_named_ent_TIME'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'TIME'))
    df['q1_named_ent_EVENT'] = df.q1_named_entities.map(lambda x: name_ent_tag(x, 'EVENT'))
    df['q2_named_ent_EVENT'] = df.q2_named_entities.map(lambda x: name_ent_tag(x, 'EVENT'))
    
    def len_share_ratio(df, wt1, wt2):
        df[wt1+'_len'] = df[wt1].map(lambda x:(len(x)))
        df[wt2+'_len'] = df[wt2].map(lambda x:(len(x)))
        ## matcher
        df[wt1+'_share'] = df.apply(lambda x:(name_ent_tag_share(x[wt1], x.q2)), axis=1)
        df[wt2+'_share'] = df.apply(lambda x:(name_ent_tag_share(x[wt2], x.q1)), axis=1)
        df[wt1+'_ratio'] = df[wt1+'_share'] / df[wt1+'_len']
        df[wt2+'_ratio'] = df[wt2+'_share'] / df[wt2+'_len']
        
    col = ['ORG', 'GPE', 'PERSON', 'NORP', 'LOC', 'DATE', 'TIME', 'EVENT']
    col = [['q1_named_ent_'+x, 'q2_named_ent_'+x] for x in col]
    
    for wt1,wt2 in col:
        len_share_ratio(df, wt1, wt2)
    
    # noun chunks
    df['q1_noun_chunks_len'] = df.q1_noun_chunks.map(lambda x:len(x))
    df['q2_noun_chunks_len'] = df.q2_noun_chunks.map(lambda x:len(x))
    df['q1_noun_chunks_share'] = df.apply(lambda x:(noun_chunks_share(x.q1_noun_chunks, x.q2)), axis=1)
    df['q2_noun_chunks_share'] = df.apply(lambda x:(noun_chunks_share(x.q2_noun_chunks, x.q1)), axis=1)
    df['q1_noun_chunks_share_ratio'] = df['q1_noun_chunks_share']/df['q1_noun_chunks_len']
    df['q2_noun_chunks_share_ratio'] = df['q2_noun_chunks_share']/df['q2_noun_chunks_len']
    
    
    col = df.dtypes[df.dtypes!='O'].index
    df = df[col]
    
    
    if 'id' in df.columns:
        if 'is_duplicate' in df.columns:
            df.drop(['is_duplicate'], axis=1, inplace=1)
        df.to_csv('~/Quora/feature/train_f002.csv.gz',
                  index=False, compression='gzip')
    else:
        df.to_csv('~/Quora/feature/test_f002.csv.gz',
                  index=False, compression='gzip')
        
    return
"""
cv, imp = utils.load_cv()
train = train.merge(cv[['id','yhat','d','q1_ori', 'q2_ori']], on='id')

df = train.sample(999)
df = df[['q1_noun_chunks', 'q2_noun_chunks', 'is_duplicate', 'yhat', 'd',
       'q1_ori', 'q2_ori']]

df['q1_first_VO'] = df.q1_wt.map(first_VO)
df['q2_first_VO'] = df.q2_wt.map(first_VO)


"""

#==============================================================================
# 
#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))





print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))



