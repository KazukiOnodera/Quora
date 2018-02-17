# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 02:29:36 2017

@author: Kazuki

nohup python -u 998-1_cv.py &

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
import time
st_time = time.time()


# setting
file_in = None
file_remove = None

loop = 1
p = 0.15
nround = 99999
nthreads = 20

X = utils.load_train(file_in, file_remove)

#==============================================================================
# logloss * down sampling
#==============================================================================

param = {'max_depth':15, 
         'eta':0.02,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
         'eval_metric':'logloss',
         'nthreads': nthreads,
         #'eval_metric':'auc',
         'objective':'binary:logistic'}

y = X.is_duplicate
col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']

lls = []
iters = []


for i in range(loop):
        
    X_, y_ = utils.down_sampling(X, y, p)
    
    sub = X_[col]
    X_.drop(col, axis=1, inplace=True)
    
    yhat, imp, ret = ex.stacking(X_, y_, param, nround, esr=30, nfold=5)
    
    ll = log_loss(y_, yhat)
    iter_ = ret['best_iter']
    
    print('998-1_cv LOGLOSS:', ll)
    print('998-1_cv best iter:', iter_)
    
    lls.append(ll)
    iters.append(iter_)


print(np.mean(lls), np.mean(iters))


sub['yhat'] = yhat
sub['d'] = abs(sub.is_duplicate - sub.yhat)
sub.sort_values('d', ascending=False, inplace=True)

print((time.time()-st_time)/60,'min')

raise Exception('stop')
#==============================================================================
# logloss * up sampling
#==============================================================================

param = {'max_depth':7, 
         'eta':0.2,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
         'eval_metric':'logloss',
         #'eval_metric':'auc',
         'objective':'binary:logistic'}


col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
y_train = X.is_duplicate

lls = []
iters = []

for i in range(loop):
    
    X_train_up, y_train_up = utils.up_sampling(X.drop(col, axis=1), y_train)

    yhat, imp, ret = ex.stacking(X_train_up, y_train_up, param, 5000, esr=30)
    
    
    ll = log_loss(y_train, yhat)
    iter_ = ret['best_iter']
    
    print('LOGLOSS:', ll)
    print('best iter:', iter_)
    
    lls.append(ll)
    iters.append(iter_)
    
    
    
"""


"""


#==============================================================================
# logloss NO sampling
#==============================================================================

param = {'max_depth':7, 
         'eta':0.2,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
         'eval_metric':'logloss',
         #'eval_metric':'auc',
         'objective':'binary:logistic'}


col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
y_train = X.is_duplicate
sub = X['id', 'is_duplicate']
X.drop(col, axis=1, inplace=1)

yhat, imp, ret = ex.stacking(X, y_train, param, 5000, esr=30)

sub['yhat'] = yhat

sub.to_csv('../output/cv.csv.gz', index=False, compression='gzip')
imp.to_csv('../output/imp.csv.gz', index=False, compression='gzip')
"""



"""
#==============================================================================
# AUC * down sampling
#==============================================================================

def calc_test_logloss(y, yhat, p=0.125):
    
    yy = pd.concat([y,yhat], axis=1)
    size = int(y.shape[0]*p)
    pos = yy[y==1].sample(size)
    neg = yy[y==0]
    #pd.concat([pos,neg], ignore_index=True).is_duplicate.mean()
    yy = pd.concat([pos,neg], ignore_index=True)
    
    print(log_loss(yy[0], yy[1]))


param = {'max_depth':7, 
         'eta':0.2,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
#         'eval_metric':'logloss',
         'eval_metric':'auc',
         'objective':'binary:logistic'}


y = X.is_duplicate

lls = []
iters = []

for i in range(3):
    X_, y_ = utils.down_sampling(X, y)
    
    col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
    sub = X_[col]
    X_.drop(col, axis=1, inplace=True)
    
    yhat, imp, ret = ex.stacking(X_, y_, param, 5000, esr=30)
    
    ll = log_loss(y_, yhat)
    iter_ = ret['best_iter']
    
    print('LOGLOSS:', ll)
    print('best iter:', iter_)
    
    lls.append(ll)
    iters.append(iter_)


print(np.mean(lls), np.mean(iters))


calc_test_logloss(y_,yhat)

"""
f011: 0.274


"""
#==============================================================================
# AUC * NO sampling
#==============================================================================


param = {'max_depth':7, 
         'eta':0.4,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
#         'eval_metric':'logloss',
         'eval_metric':'auc',
         'objective':'binary:logistic'}


y = X.is_duplicate


col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
sub = X[col]
X.drop(col, axis=1, inplace=True)

lls = []
iters = []
for i in range(3):
    
    
    yhat, imp, ret = ex.stacking(X, y, param, 5000, esr=30)
    
    ll = log_loss(y, yhat)
    iter_ = ret['best_iter']
    
    print('LOGLOSS:', ll)
    print('best iter:', iter_)
    
    lls.append(ll)
    iters.append(iter_)


print(np.mean(lls), np.mean(iters))

"""
f011: 0.9025


"""


    
