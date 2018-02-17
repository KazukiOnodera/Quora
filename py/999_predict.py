# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 03:23:59 2017

@author: Kazuki

nohup python -u 999_predict.py &


"""

import xgboost as xgb
from sklearn.metrics import log_loss
import gc
import utils
import numpy as np
import time
st_time = time.time()

try:
    import xgbextension as ex
except:
    import sys
    sys.path.append('/home/konodera/Python')
    import xgbextension as ex

# setting
loop = 1
nround = 1200
file_in = None
file_remove = None
is_mirror = 1

date = '423'

fname = '../output/{0}_1_f014_r{1}.csv.gz'.format(date, nround)

#==============================================================================
# train
#==============================================================================

param = {'max_depth':15, 
         'eta':0.02,
         'colsample_bytree':0.6,
         'subsample':0.5,
         'silent':1, 
         'scale_pos_weight':0.37,
         'eval_metric':'logloss',
         'objective':'binary:logistic'}

train = utils.load_train(file_in, file_remove)

col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
y_train = train.is_duplicate
train.drop(col, axis=1, inplace=True)

print('train shape:{}'.format(train.shape))

dtrain = xgb.DMatrix(train, label=y_train)

models = []

for i in range(loop):
    
    print('loop:{}'.format(i))
    
    param.update({'seed':np.random.randint(999999)})
    model = xgb.train(param, dtrain, nround)
    yhat = model.predict(dtrain)
    print('LOGLOSS:',log_loss(y_train, yhat))
    
    models.append(model)
    model.save_model('../model/xgb{}.model'.format(i))

train_col = dtrain.feature_names
del train, dtrain, y_train
gc.collect()

imp = ex.getImp(models)
imp.to_csv('../output/imp-{}.csv'.format(date), index=0)

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



