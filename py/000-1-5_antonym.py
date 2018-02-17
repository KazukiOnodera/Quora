# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:08:41 2017

@author: konodera
"""


import os
from collections import Counter
import pandas as pd
from itertools import product
import utils
stops = utils.stops
w2v_google = utils.load_vec('w2v-google')
#w2v_quora = utils.load_vec('w2v-quora-0')

# param
sim_value = 0.5


train = pd.merge(pd.read_pickle('../input/mk/train_and-or_0.p'),
                 pd.read_csv('../input/train.csv.zip', usecols=['id','is_duplicate']),
                 on='id', how='left')


#==============================================================================
# def
#==============================================================================
def most_sim(s1, s2):
    """
    """
    s1 = utils.valid_words(w2v_google, s1-stops)
    s2 = utils.valid_words(w2v_google, s2-stops)
    words_sets = product(s1,s2)
    sims = [sorted([w1,w2]) for w1,w2 in words_sets if w2v_google.similarity(w1,w2)>sim_value]
#    sims = [(w1,w2) for w1,w2 in words_sets if w2v_google.similarity(w1,w2)>sim_value or w2v_quora.similarity(w1,w2)>sim_value]
    return sims
#==============================================================================

train['set_diff_most-sim'] = train.apply(lambda x: most_sim(x.q1_set_diff, x.q2_set_diff), axis=1)
train = train[['id', 'q1', 'q2', '_and_', 'set_diff', 'is_duplicate', 'set_diff_most-sim']]


train_ = train[train['set_diff_most-sim'].map(len)>0]






synonyms = sum(train.ix[train.is_duplicate==1]['set_diff_most-sim'].tolist(),[])
synonyms = [' <-> '.join([x[0],x[1]]) for x in synonyms]
#synonyms = [' <-> '.join([utils.pt(x[0]),utils.pt(x[1])]) for x in synonyms]
synonyms = Counter(synonyms)



antonyms = sum(train.ix[train.is_duplicate==0]['set_diff_most-sim'].tolist(),[])
antonyms = [' <-> '.join([x[0],x[1]]) for x in antonyms]
#antonyms = [' <-> '.join([utils.pt(x[0]),utils.pt(x[1])]) for x in antonyms]
antonyms = Counter(antonyms)


df = pd.DataFrame(list(synonyms.items()),
                  columns=['word', 'cnt_1']).set_index('word')

df_ = pd.DataFrame(list(antonyms.items()),
                  columns=['word', 'cnt_0']).set_index('word')

df = pd.concat([df,df_], axis=1)
df_ = df[df.cnt_1.isnull()]

os.system("say 'done'")




print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

