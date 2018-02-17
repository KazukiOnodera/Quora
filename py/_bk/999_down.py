# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 03:23:59 2017

@author: Kazuki

nohup python -u 999_down.py &


"""

import xgboost as xgb
from sklearn.metrics import log_loss
import gc
import utils
import numpy as np
import time
st_time = time.time()

# setting
loop = 3
nround = 4000
p = 0.15
file_in = None
file_remove = None
is_mirror = 1



fname = '../output/408_1_f011_down_p{0}_r{1}.csv.gz'.format(p,nround)

#==============================================================================
# train
#==============================================================================

param = {'max_depth':15, 
         'eta':0.02,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
         'eval_metric':'logloss',
         'objective':'binary:logistic'}

train = utils.load_train(file_in, file_remove)

col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
y_train = train.is_duplicate
train.drop(col, axis=1, inplace=True)


models = []

for i in range(loop):
    
    print('loop:{}'.format(i))
    
    X_train_down, y_train_down = utils.down_sampling(train, y_train, p)
    dtrain = xgb.DMatrix(X_train_down, label=y_train_down)
    
    param.update({'seed':np.random.randint(999999)})
    model = xgb.train(param, dtrain, nround)
    yhat = model.predict(dtrain)
    print('LOGLOSS:',log_loss(y_train_down, yhat))
    
    models.append(model)

train_col = dtrain.feature_names
del train, dtrain, X_train_down, y_train_down
gc.collect()
#==============================================================================
# test
#==============================================================================

test1 = utils.load_test(file_in, file_remove)

col = ['test_id', 'question1', 'question2']
sub = test1[col]
test1.drop(col, axis=1, inplace=1)


if is_mirror:
    print('q1_to_q2!')
    test2 = utils.q1_to_q2(test1)

dtest1 = xgb.DMatrix(test1[train_col])
dtest2 = xgb.DMatrix(test2[train_col])
del test1,test2; gc.collect()


sub['is_duplicate'] = 0
for model in models:
    sub['is_duplicate'] += model.predict(dtest1)
    sub['is_duplicate'] += model.predict(dtest2)
sub['is_duplicate'] /=(loop*2)

sub.ix[sub.question1.isnull(), 'is_duplicate'] = 0
sub.ix[sub.question2.isnull(), 'is_duplicate'] = 0

print((time.time()-st_time)/60,'min')
print('pred mean:',sub.is_duplicate.mean())

#raise

sub[['test_id','is_duplicate']].to_csv(fname,
               index=False, compression='gzip')

print("="*50)
print('fname:', fname)
print("="*50)


"""

import xgbextension as ex
imp = ex.getImp(models[1])









"""



