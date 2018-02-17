# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:45:55 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

#import pandas as pd
#import numpy as np
import utils
utils.load_vec()
import multiprocessing as mp
total_proc = 2


train, test = utils.load(2)



#==============================================================================
# def
#==============================================================================
def alpha(s):
    return [w.lower() for w in s.split() if w.isalpha()]
    
def main(df):
    """
    df = train.sample(999)
    """
    
    df.q1 = df.q1.map(alpha)
    df.q2 = df.q2.map(alpha)
    
    df['sen_sim_d2v_wiki'] = df.apply(lambda x: utils.d2v_similarity(x.q1, x.q2,'wiki'), axis=1)
    df['sen_sim_d2v_apn'] = df.apply(lambda x: utils.d2v_similarity(x.q1, x.q2,'apnews'), axis=1)
    df['sen_sim_d2v_quora'] = df.apply(lambda x: utils.d2v_similarity(x.q1, x.q2,'quora'), axis=1)
    
    df['sen_sim_w2v_ggl'] = df.apply(lambda x: utils.w2v_similarity(x.q1, x.q2,'google'), axis=1)
    df['sen_sim_w2v_wiki'] = df.apply(lambda x: utils.w2v_similarity(x.q1, x.q2,'wiki'), axis=1)
    df['sen_sim_w2v_apn'] = df.apply(lambda x: utils.w2v_similarity(x.q1, x.q2,'apnews'), axis=1)
    df['sen_sim_w2v_quora'] = df.apply(lambda x: utils.w2v_similarity(x.q1, x.q2,'quora'), axis=1)
    
    df['sen_sim_glv_840'] = df.apply(lambda x: utils.glv_similarity(x.q1, x.q2,'840'), axis=1)
    df['sen_sim_glv_twi'] = df.apply(lambda x: utils.glv_similarity(x.q1, x.q2,'twitter'), axis=1)
    
    
    col = df.dtypes[df.dtypes!='object'].index.tolist()
    
    path = '~/Quora/feature/'
    name = 'f012'
    if 'id' in df.columns:
        df[col].to_csv(path+'train_{0}.csv.gz'.format(name), index=False, compression='gzip')
    else:
        df[col].to_csv(path+'test_{0}.csv.gz'.format(name), index=False, compression='gzip')
    
    return

#==============================================================================
# 
#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, [train, test])



print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

