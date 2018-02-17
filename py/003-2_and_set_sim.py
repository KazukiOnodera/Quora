# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:40:31 2017

@author: Kazuki
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import utils
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
import dask.dataframe as dd

#==============================================================================
# setting
#==============================================================================
total_proc = 10

n = 10


print('load train')
train = pd.merge(pd.read_csv('../feature/train_f003_2_intersect.csv.gz'),
                 pd.read_csv('../feature/train_f003_2_set_diff.csv.gz'),
                 on='id')
train = utils.merge_y(train)

print('load test')
test = pd.merge(pd.read_csv('../feature/test_f003_2_intersect.csv.gz'),
                 pd.read_csv('../feature/test_f003_2_set_diff.csv.gz'),
                 on='test_id')


vec_train = np.array(train)

#==============================================================================
# def
#==============================================================================
def most_sim_exp_leave_one_out(vec):
    """
    return top n exp
    """
    
    sim = cosine_similarity(vec_train[0:,1:-1], vec[1:-1])
    sim = sim.reshape(sim.shape[0])
    ixs = sorted(np.argsort(sim)[::-1][1:n+1])
    y_ = vec_train[ixs][0:,-1]
    sims = sim[ixs]
    
    return np.mean(y_ * sims)



def most_sim_exp(vec):
    """
    return top n exp
    """
    
    sim = cosine_similarity(vec_train[0:,1:-1], vec[1:-1])
    sim = sim.reshape(sim.shape[0])
    ixs = sorted(np.argsort(sim)[::-1][:n])
    y_ = vec_train[ixs][0:,-1]
    sims = sim[ixs]
    
    return np.mean(y_ * sims)





def debug():

    train_ori = utils.merge_y(pd.read_pickle('../input/mk/train_and-or_2.p')) 
    train_ori.drop(['q1_set_diff', 'q2_set_diff', '_or_'], axis=1, inplace=1)
    
    train_id = 237672
    vec = np.array(train.ix[train.id==train_id])
    print(train_ori.ix[train_ori.id==vec[0,0]])
    
    n = 10
    sim = cosine_similarity(vec_train[0:,1:-1], vec[0,1:-1])
    sim = sim.reshape(sim.shape[0])
    
    ixs = sorted(np.argsort(sim)[::-1][:n]) # include all
    sims = sim[ixs]
    
    ids = vec_train[ixs][0:, 0]
    
    tmp = train_ori.ix[train_ori.id.isin(ids)]
    tmp['sim'] = sims
    (tmp.is_duplicate * tmp.sim).mean()

    
def main(df):
    """
    
    df = train.sample(999)
    
    """
    
    
    ddf = dd.from_pandas(df, total_proc)
    
    if 'id' in df.columns:
        df['and_set_sim'] = ddf.apply(most_sim_exp_leave_one_out, axis=1).compute()
    else:
        df['and_set_sim'] = ddf.apply(most_sim_exp, axis=1).compute()
    
    path = '~/Quora/feature/'
    name = 'f003_and_set_sim'
    if 'id' in df.columns:
        df[['id', 'and_set_sim']].to_csv(path+'train_{0}.csv.gz'.format(name), index=False, compression='gzip')
    else:
        df[['test_id', 'and_set_sim']].to_csv(path+'test_{0}.csv.gz'.format(name), index=False, compression='gzip')
    


#==============================================================================
# main
#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))





















