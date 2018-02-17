# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:03:06 2017

@author: Kazuki


https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb/run/1015279

nohup python 101_from_anokas.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
import numpy as np
import gc
import utils

#==============================================================================
# def
#==============================================================================
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['q1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['q2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['q1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['q2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def main(n, suf):
    global weights
    
    train, test = utils.load(n)

    train_qs = pd.Series(train['q1'].tolist() + train['q2'].tolist()).astype(str)
    test_qs = pd.Series(test['q1'].tolist() + test['q2'].tolist()).astype(str)
    # delete dup
    qs = pd.concat([train_qs, test_qs]).drop_duplicates().reset_index(drop=1)
    
    
#    eps = 5000 
#    words = (" ".join(train_qs)).lower().split()
    #
#    words = (" ".join(train_qs)).lower().split() + (" ".join(test_qs)).lower().split()
    words = (" ".join(qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    
    # mk train
    train['word_match_share_anokas'+suf] = train.apply(word_match_share, axis=1, raw=True)
    train['tfidf_word_match_anokas'+suf] = train.apply(tfidf_word_match_share, axis=1, raw=True)
    
    train.drop(['q1', 'q2'], axis=1, inplace=True)
    train.to_csv('../feature/train_f101-{0}.csv.gz'.format(n), 
                 index=False, compression='gzip')
    del train; gc.collect()
    
    # mk test
    test['word_match_share_anokas'+suf] = test.apply(word_match_share, axis=1, raw=True)
    test['tfidf_word_match_anokas'+suf] = test.apply(tfidf_word_match_share, axis=1, raw=True)
    
    test.drop([ 'q1', 'q2'], axis=1, inplace=True)
    test.to_csv('../feature/test_f101-{0}.csv.gz'.format(n), 
                index=False, compression='gzip')
    del test; gc.collect()

#==============================================================================

main(0, '')
main(1, '-stem')
main(2, '-stop')
main(3, '-stst')
#main(9, '-syn1')




print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))



