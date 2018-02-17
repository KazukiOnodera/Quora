# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 02:29:36 2017

@author: Kazuki

nohup python -u 998-2_cv_all.py & 

"""

import pandas as pd
try:
    import xgbextension as ex
except:
    import sys
    sys.path.append('/home/konodera/Python')
    import xgbextension as ex

from sklearn.metrics import log_loss
import utils
import numpy as np


# setting
file_remove = ['f009-word', 'f013_disjunction']
nround = 99999

loop = 1


X = utils.load_train(file_remove=file_remove)


#==============================================================================
# logloss NO sampling
#==============================================================================

param = {'max_depth':15, 
         'eta':0.02,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
         'eval_metric':'logloss',
         'scale_pos_weight':0.37,
#         'scale_pos_weight':1.707, # neg/pos
         'objective':'binary:logistic'}


col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
y_train = X.is_duplicate
sub = X[['id', 'is_duplicate']]
X.drop(col, axis=1, inplace=1)

def get_valid_col(col):
    return [c for c in col if c.count(',')>0 or c.count('[')>0 or c.count(']')>0 or c.count('>')>0]
col = get_valid_col(X.columns)
X.drop(col, axis=1, inplace=1)

yhat, imp, ret = ex.stacking(X, y_train, param, nround, esr=30)

sub['yhat'] = yhat

sub[['id', 'yhat']].to_csv('../output/cv.csv.gz', index=False, compression='gzip')
imp.to_csv('../output/imp.csv.gz', index=False, compression='gzip')


print(log_loss(y_train, yhat))

"""
eta: 0.02
CV: 0.317928303359 ==> LB: 0.28018

eta: 0.2
CV: 0.363839232148


"""





