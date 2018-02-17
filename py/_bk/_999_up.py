# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 03:23:59 2017

@author: Kazuki

nohup 999_up.py &


"""


import pandas as pd
from glob import glob
import xgboost as xgb
from sklearn.metrics import log_loss
import gc
import utils
import numpy as np
import time
st_time = time.time()

# setting
loop = 3
nround = 3500

param = {'max_depth':7, 
         'eta':0.2,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
         'eval_metric':'logloss',
         'objective':'binary:logistic'}


train = utils.load_train()

col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
y_train = train.is_duplicate
X_train, y_train = utils.up_sampling(train.drop(col, axis=1), y_train)
dtrain = xgb.DMatrix(X_train, label=y_train)


    
models = []

for i in range(loop):
    param.update({'seed':np.random.randint(999999)})
    model = xgb.train(param, dtrain, nround)
    yhat = model.predict(dtrain)
    print('LOGLOSS:',log_loss(y_train, yhat))
    
    models.append(model)

del dtrain, X_train, y_train
gc.collect()
#==============================================================================
# 
#==============================================================================
test = utils.load_test()

gc.collect()

col = ['test_id', 'question1', 'question2']
sub = test[col]
dtest = xgb.DMatrix(test.drop(col, axis=1))


    
sub['is_duplicate'] = 0
for model in models:
    sub['is_duplicate'] += model.predict(dtest)
sub['is_duplicate'] /=loop

sub.ix[test.question1.isnull(), 'is_duplicate'] = 0
sub.ix[test.question2.isnull(), 'is_duplicate'] = 0

print((time.time()-st_time)/60,'min')
print('pred mean:',sub.is_duplicate.mean())

#raise


sub[['test_id','is_duplicate']].to_csv('../output/327-3_f008_up_r{0}.csv.gz'.format(nround),
               index=False, compression='gzip')



