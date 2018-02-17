# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:38:32 2017

@author: Kazuki

nohup ipython 000-0_mk_stem_stop.py &

"""

import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
from nltk import stem
pt = stem.PorterStemmer()
import utils
stops = utils.stops
stops_kigou = set(list(stops)+list(utils.kigou))
import multiprocessing as mp
total_proc = 2


#==============================================================================
# preprocess
#==============================================================================

train = pd.read_csv('../input/train.csv.zip')
test  = pd.read_csv('../input/test.csv.zip')


train['q1'] = train.question1.map(utils.preprocessing)
train['q2'] = train.question2.map(utils.preprocessing)
test['q1']  = test.question1.map(utils.preprocessing)
test['q2']  = test.question2.map(utils.preprocessing)


# drop
col = ['q1', 'q2']
train = train[['id']+col]
test  = test[['test_id']+col]


train.to_csv('../input/mk/train_prep.csv.gz',
             compression='gzip', index=False)
test.to_csv('../input/mk/test_prep.csv.gz',
             compression='gzip', index=False)

#==============================================================================
# def
#==============================================================================
def get_stem(txt):
    try:
        return ' '.join(list(map(pt.stem, txt.split()))).strip()
    except:
        print(txt)
        return txt

def rem_stops(txt):
    try:
        return ' '.join([w for w in txt.lower().split() if w not in stops_kigou]).strip()
    except:
        print(txt)
        return txt

def multi(p):
    if p==0:
        #==============================================================================
        print('start stem')
        #==============================================================================
        train, test = utils.load(0)
        
        train['q1'] = train['q1'].map(get_stem)
        train['q2'] = train['q2'].map(get_stem)
        test['q1']  = test['q1'].map(get_stem)
        test['q2']  = test['q2'].map(get_stem)
        
        # drop
        col = ['q1', 'q2']
        train = train[['id']+col]
        test  = test[['test_id']+col]
        
        
        train.to_csv('../input/mk/train_stem.csv.gz',
                     compression='gzip', index=False)
        test.to_csv('../input/mk/test_stem.csv.gz',
                     compression='gzip', index=False)
        
    elif p == 1:
        #==============================================================================
        print('start stops')
        #==============================================================================
        
        train, test = utils.load(0)
        
        train['q1'] = train['q1'].map(rem_stops)
        train['q2'] = train['q2'].map(rem_stops)
        test['q1']  = test['q1'].map(rem_stops)
        test['q2']  = test['q2'].map(rem_stops)
        
        
        # drop
        col = ['q1', 'q2']
        train = train[['id']+col]
        test  = test[['test_id']+col]
        
        
        train.to_csv('../input/mk/train_stop.csv.gz',
                     compression='gzip', index=False)
        test.to_csv('../input/mk/test_stop.csv.gz',
                     compression='gzip', index=False)
        
        #==============================================================================
        print('start stop & stem')
        #==============================================================================
        train['q1'] = train['q1'].map(get_stem)
        train['q2'] = train['q2'].map(get_stem)
        test['q1']  = test['q1'].map(get_stem)
        test['q2']  = test['q2'].map(get_stem)
        
        # drop
        col = ['q1', 'q2']
        train = train[['id']+col]
        test  = test[['test_id']+col]
        
        
        train.to_csv('../input/mk/train_stop-stem.csv.gz',
                     compression='gzip', index=False)
        test.to_csv('../input/mk/test_stop-stem.csv.gz',
                     compression='gzip', index=False)
        
    return
#==============================================================================
# main
#==============================================================================
pool = mp.Pool(total_proc)
callback = pool.map(multi, range(total_proc))


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))


