# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:05:04 2017

@author: konodera
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
total_proc = 2
import utils

#w2v_google = utils.load_vec('w2v-google')
w2v_quora = utils.load_vec('w2v-quora-2')


train, test = utils.load(9, 1)

#==============================================================================
# def
#==============================================================================

def main(p):
    """
    df = train.sample(9999)
    
    df.drop(['q1_set_diff','q2_set_diff'],axis=1,inplace=1)
    
    w = 'weight'
    df_ = df[df.intersect.map(lambda x: w in x)]
    df_ = df[df.set_diff.map(lambda x: w in x)]
    
    
    
    
    """
    if p==0:
        df = train
    elif p==1:
        df = test
    
    # sen2vec
    def to_vec(sentenses):
        vec_li = [utils.sen2vec(w2v_quora, words, 100) for words in sentenses]
        vec = np.array(vec_li)
        return vec
        
    vec_intersect = to_vec(df['_and_'].tolist())
    vec_set_diff  = to_vec(df['set_diff'].tolist())
#    vec_disjunction = to_vec(df['disjunction'].tolist())
    
    if 'id' in df.columns:
        # intersect
        df_ = df[['id']].reset_index(drop=1)
        df_ = pd.concat([df_, pd.DataFrame(vec_intersect)],axis=1)
        df_.columns = ['id']+['f003_2_intersect-{}'.format(c) for c in df_.columns[1:]]
        df_.to_csv('../feature/train_f003_2_intersect.csv.gz',index=0,compression='gzip')
        
        # set_diff
        df_ = df[['id']].reset_index(drop=1)
        df_ = pd.concat([df_, pd.DataFrame(vec_set_diff)],axis=1)
        df_.columns = ['id']+['f003_2_set_diff-{}'.format(c) for c in df_.columns[1:]]
        df_.to_csv('../feature/train_f003_2_set_diff.csv.gz',index=0,compression='gzip')
        
        # disjunction
#        df_ = df[['id']].reset_index(drop=1)
#        df_ = pd.concat([df_, pd.DataFrame(vec_disjunction)],axis=1)
#        df_.columns = ['id']+['f003_2_disjunction-{}'.format(c) for c in df_.columns[1:]]
#        df_.to_csv('../feature/train_f003_2_disjunction.csv.gz',index=0,compression='gzip')
    else:
        # intersect
        df_ = df[['test_id']].reset_index(drop=1)
        df_ = pd.concat([df_, pd.DataFrame(vec_intersect)],axis=1)
        df_.columns = ['test_id']+['f003_2_intersect-{}'.format(c) for c in df_.columns[1:]]
        df_.to_csv('../feature/test_f003_2_intersect.csv.gz',index=0,compression='gzip')
        
        # set_diff
        df_ = df[['test_id']].reset_index(drop=1)
        df_ = pd.concat([df_, pd.DataFrame(vec_set_diff)],axis=1)
        df_.columns = ['test_id']+['f003_2_set_diff-{}'.format(c) for c in df_.columns[1:]]
        df_.to_csv('../feature/test_f003_2_set_diff.csv.gz',index=0,compression='gzip')
        
        # disjunction
#        df_ = df[['test_id']].reset_index(drop=1)
#        df_ = pd.concat([df_, pd.DataFrame(vec_disjunction)],axis=1)
#        df_.columns = ['test_id']+['f013_disjunction-{}'.format(c) for c in df_.columns[1:]]
#        df_.to_csv('../feature/test_f013_disjunction.csv.gz',index=0,compression='gzip')
    
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

