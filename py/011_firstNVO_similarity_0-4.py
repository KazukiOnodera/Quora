# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:04:47 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))


import utils
utils.load_vec()
stops = utils.stops
import pandas as pd
import gc
import multiprocessing as mp
total_proc = 2



train, test = utils.load(4)

train_, test_ = utils.load(0)

train = pd.merge(train, train_, on='id', how='left')
test  = pd.merge(test, test_, on='test_id', how='left')

del train_, test_; gc.collect()


#==============================================================================
# def
#==============================================================================

def first_N(wt):
    for w,t in wt:
        if w.lower() in stops:
            continue
        if t.startswith('N'):
            return w
    return

def first_VO(wt):
    v = None
    o = []
    sw = False
    for w,t in wt:
        
        if sw and t.startswith('V'):
#            print(v,w)
            v = w
            sw = False
            continue
        elif sw and w =='to':
            continue
        else:
            sw = False
        
        # remove stop words
        if v is None and len(o)==0 and w.lower() in stops:
            # v == None, o == []
            continue
        elif v is not None and len(o)==0 and w == 'in':
            # v == 'VB', o == []
            break
        elif v is not None and len(o)==0 and w.lower() in stops:
            # v == 'VB', o == []
            continue
        elif v is not None and len(o)>0 and w.lower() in stops:
            # v == 'VB', o == ['N']
            break
        
        # first verb
        if v is None and t.startswith('V'):
            v = w
            sw = True
            continue
        
        # objective
        if v is not None and (t.startswith('N') or t.startswith('J')):
            o.append(w)
        elif v is not None and (t.startswith('VBN')):
            o.append(w) # heal your broken heart
        elif v is not None and not (t.startswith('N') or t.startswith('J')):
            break
        
    return v,o

def main(p):
    """
    df = train.head(999)
    """
    if p==0:
        df = train
    elif p==1:
        df = test
    
    df['q1_first_N'] = df.q1_wt.map(first_N)
    df['q2_first_N'] = df.q2_wt.map(first_N)
    df['N_match'] = (df['q1_first_N']==df['q2_first_N'])*1
    
    df['N_similarity_d2v_wiki'] = df.apply(lambda x: utils.d2v_similarity([x.q1_first_N], [x.q2_first_N],'wiki'), axis=1)
    df['N_similarity_d2v_apn'] = df.apply(lambda x: utils.d2v_similarity([x.q1_first_N], [x.q2_first_N],'apnews'), axis=1)
    df['N_similarity_d2v_quora'] = df.apply(lambda x: utils.d2v_similarity([x.q1_first_N], [x.q2_first_N],'quora'), axis=1)

    df['N_similarity_w2v_ggl'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_N], [x.q2_first_N],'google'), axis=1)
    df['N_similarity_w2v_wiki'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_N], [x.q2_first_N],'wiki'), axis=1)
    df['N_similarity_w2v_apn'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_N], [x.q2_first_N],'apnews'), axis=1)
    df['N_similarity_w2v_quora'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_N], [x.q2_first_N],'quora'), axis=1)
    
    df['N_similarity_glv_840'] = df.apply(lambda x: utils.glv_similarity([x.q1_first_N], [x.q2_first_N],'840'), axis=1)
    df['N_similarity_glv_twi'] = df.apply(lambda x: utils.glv_similarity([x.q1_first_N], [x.q2_first_N],'twitter'), axis=1)
    
    
    
    
    df['q1_first_VO'] = df.q1_wt.map(first_VO)
    df['q2_first_VO'] = df.q2_wt.map(first_VO)
    df['V_match'] = (df['q1_first_VO'].map(lambda x:x[0]) == df['q2_first_VO'].map(lambda x:x[0]))*1
    
    df['V_similarity_d2v_wiki'] = df.apply(lambda x: utils.d2v_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'wiki'), axis=1)
    df['V_similarity_d2v_apn'] = df.apply(lambda x: utils.d2v_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'apnews'), axis=1)
    df['V_similarity_d2v_quora'] = df.apply(lambda x: utils.d2v_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'quora'), axis=1)
    
    df['V_similarity_d2v_ggl'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'google'), axis=1)
    df['V_similarity_w2v_wiki'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'wiki'), axis=1)
    df['V_similarity_w2v_apn'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'apnews'), axis=1)
    df['V_similarity_w2v_quora'] = df.apply(lambda x: utils.w2v_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'quora'), axis=1)
    
    df['V_similarity_glv_840'] = df.apply(lambda x: utils.glv_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'840'), axis=1)
    df['V_similarity_glv_twi'] = df.apply(lambda x: utils.glv_similarity([x.q1_first_VO[0]], [x.q2_first_VO[0]],'twitter'), axis=1)
    
    
    
    df['O_similarity_d2v_wiki'] = df.apply(lambda x: utils.d2v_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'wiki'), axis=1)
    df['O_similarity_d2v_apn'] = df.apply(lambda x: utils.d2v_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'apnews'), axis=1)
    df['O_similarity_d2v_quora'] = df.apply(lambda x: utils.d2v_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'quora'), axis=1)
    
    df['O_similarity_w2v_ggl'] = df.apply(lambda x: utils.w2v_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'google'), axis=1)
    df['O_similarity_w2v_wiki'] = df.apply(lambda x: utils.w2v_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'wiki'), axis=1)
    df['O_similarity_w2v_apn'] = df.apply(lambda x: utils.w2v_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'apnews'), axis=1)
    df['O_similarity_w2v_quora'] = df.apply(lambda x: utils.w2v_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'quora'), axis=1)
    
    df['O_similarity_glv_840'] = df.apply(lambda x: utils.glv_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'840'), axis=1)
    df['O_similarity_glv_twi'] = df.apply(lambda x: utils.glv_similarity(x.q1_first_VO[1], x.q2_first_VO[1],'twitter'), axis=1)
    
    col = df.dtypes[df.dtypes!='object'].index.tolist()
    
    path = '~/Quora/feature/'
    name = 'f011'
    if 'id' in df.columns:
        df[col].to_csv(path+'train_{0}.csv.gz'.format(name), index=False, compression='gzip')
    else:
        df[col].to_csv(path+'test_{0}.csv.gz'.format(name), index=False, compression='gzip')
    
    return
    
"""

s = df.sample(1).q1.values[0].split()

for w1,w2 in zip(s,s[1:]):
    print(w1,w2,model.similarity(w1, w2))


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


