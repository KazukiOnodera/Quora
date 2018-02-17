# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:14:36 2017

@author: konodera
"""

import pandas as pd
import utils
from sklearn.feature_extraction.text import TfidfVectorizer

train, test = utils.load(8, 1)

train = train[['id', '_and_', 'set_diff']]
test  = test[['test_id', '_and_', 'set_diff']]


def to_str(df):
    df['_and_'] = df['_and_'].map(lambda x: ' '.join(x))
    df['set_diff'] = df['set_diff'].map(lambda x: ' '.join(x))
    return df

train = to_str(train)
test = to_str(test)


#==============================================================================
# and
#==============================================================================
clf = TfidfVectorizer(max_df=0.5, min_df=1, max_features=100, norm='l2')

clf.fit(train._and_.tolist() + test._and_.tolist())

train_tfidf = clf.transform(train._and_.tolist())
test_tfidf  = clf.transform(test._and_.tolist())


train_ = train[['id']].reset_index(drop=1)
train_ = pd.concat([train_, pd.DataFrame(train_tfidf.toarray())],axis=1)
train_.columns = ['id']+['f003_and_tfidf-{}'.format(c) for c in train_.columns[1:]]
train_.to_csv('../feature/train_f003_and_tfidf.csv.gz',index=0,compression='gzip')

test_ = test[['test_id']].reset_index(drop=1)
test_ = pd.concat([test_, pd.DataFrame(test_tfidf.toarray())],axis=1)
test_.columns = ['test_id']+['f003_and_tfidf-{}'.format(c) for c in test_.columns[1:]]
test_.to_csv('../feature/test_f003_and_tfidf.csv.gz',index=0,compression='gzip')



#==============================================================================
# set-diff
#==============================================================================
clf = TfidfVectorizer(max_df=0.5, min_df=1, max_features=100, norm='l2')

clf.fit(train.set_diff.tolist() + test.set_diff.tolist())

train_tfidf = clf.transform(train.set_diff.tolist())
test_tfidf  = clf.transform(test.set_diff.tolist())


train_ = train[['id']].reset_index(drop=1)
train_ = pd.concat([train_, pd.DataFrame(train_tfidf.toarray())],axis=1)
train_.columns = ['id']+['f003_set-diff_tfidf-{}'.format(c) for c in train_.columns[1:]]
train_.to_csv('../feature/train_f003_set-diff_tfidf.csv.gz',index=0,compression='gzip')

test_ = test[['test_id']].reset_index(drop=1)
test_ = pd.concat([test_, pd.DataFrame(test_tfidf.toarray())],axis=1)
test_.columns = ['test_id']+['f003_set-diff_tfidf-{}'.format(c) for c in test_.columns[1:]]
test_.to_csv('../feature/test_f003_set-diff_tfidf.csv.gz',index=0,compression='gzip')






print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

