# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:19:05 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
try:
    import xgbextension as ex
except:
    import sys
    sys.path.append('/home/konodera/Python')
    import xgbextension as ex
import numpy as np
import gc
from sklearn.metrics import log_loss
import utils


# setting
file_in = ['f103']
nround = 99999

loop = 1


param = {'max_depth':15, 
         'eta':0.1,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
#         'scale_pos_weight':1.707, # neg/pos
         'eval_metric':'auc',
         'objective':'binary:logistic'}

train = utils.load_train(file_in=file_in)
test =  utils.load_test(file_in=file_in)

#==============================================================================
# logloss NO sampling
#==============================================================================
def get_valid_col(col):
    return [c for c in col if c.count(',')>0 or c.count('[')>0 or c.count(']')>0 or c.count('>')>0]

col = ['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
y_train = train.is_duplicate
train_sub = train[['id', 'is_duplicate']]
train.drop(col, axis=1, inplace=1) # keep id for merge 010

col = ['question1', 'question2']
test.drop(col, axis=1, inplace=1) # keep test_id for merge 010
test_sub = test[['test_id']]

for i in range(1):
    train = pd.merge(train, utils.load_010([0,1,2],1), on='id', how='left')
    test  = pd.merge(test,  utils.load_010([0,1,2],0), on='test_id', how='left')
    
    col = get_valid_col(train.columns)
    train.drop(col, axis=1, inplace=1)
    test.drop(col, axis=1, inplace=1)
    
    yhat, imp, ret = ex.stacking(train.drop('id', axis=1), y_train, param, nround, 
                                 esr=30, test=test.drop('test_id', axis=1))
    gc.collect()
    
    print('log_loss:',log_loss(y_train, yhat))
    train_sub['yhat'] = yhat
    test_sub['yhat']  = ret.get('test')

train_sub.drop('is_duplicate', axis=1).to_csv('../feature/train_f009_xgb.csv.gz', index=False, compression='gzip')
train_sub.to_csv('../feature/test_f009_xgb.csv.gz', index=False, compression='gzip')






print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

