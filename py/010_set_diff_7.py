# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:08:23 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
import numpy as np
import pandas as pd
from nltk import stem
pt = stem.PorterStemmer().stem

from gensim.models import Doc2Vec
d2v = Doc2Vec.load('../nlp_source/d2v/enwiki_dbow/doc2vec.bin')
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)

import gc


train, test = utils.load(7, 1)

train_, test_ = utils.load(2)
train = pd.merge(train, train_, on='id', how='left')
test  = pd.merge(test, test_, on='test_id', how='left')
del train_, test_; gc.collect()


tf = utils.load_tf()

"""
model.similarity('mozzarella', 'cheeze')

"""

#==============================================================================
# def
#==============================================================================
def set_diff(wt1, wt2):
    try:
        s2_ = [pt(w2.lower()) for w2,t2 in wt2]
        return [(w1,t1) for w1,t1 in wt1 if pt(w1.lower()) not in s2_]
    except:
        print(wt1, wt2)
        return []

def count_tag(wt, tag):
    return sum([1 for w,t in wt if t.startswith(tag)])

def most_sim(wt, s):
    """
    return: most high similarity list
    
    exp1:
    words = ['app', 'protection']
    s = 'best smartphone security apps'
    
    exp2:
    words = ['factors', 'affect']
    s ='determine decomposition copper nitrate'
    return [0.22852131824433961, 0.28510875898648874]
    
    exp3:
    words = ['possible', 'future', 'traval']
    s = 'time travel'
    
    exp4:
    words = ['way', 'learn', 'best']
    s = 'start learning hacking'
    
    exp5:
    words = ['patagonian']
    s = 'cold gobi desert get average temperatures compare ones colorado plateau'
    
    """
    if not isinstance(s, str) or wt == []:
        return [-1]
    
    words = [w for w,t in wt if w.isalpha()]
    s = [w for w in s.split() if w.isalpha()]
#    weights = [tf.get(w,0) for w in words]
#    weights_sum = sum(weights)
#    weights = [w/weights_sum for w in weights]
    
    ret = []
    for wo in words:
        sims = []
        for w_ in s:
            try:
                sims.append(w2v.similarity(wo, w_))
            except:
                # for not found
                pass
        if len(sims)>0:
            ret.append(max(sims))
    return ret

def get_weights(words):
    return [tf.get(w,0) for w,t in words]



def main(df):
    """
    cv, imp = utils.load_cv()
    train = train.merge(cv[['id','yhat','d','q1_ori', 'q2_ori']], on='id')
    
    df = train.sample(999)
    
    df['q1_set_diff'] = df.apply(lambda x: set_diff(x.q1_wt, x.q2_wt), axis=1) # for broadcast error
    df['q2_set_diff'] = df.apply(lambda x: set_diff(x.q2_wt, x.q1_wt), axis=1)
    
    df = df[['q1_ori', 'q2_ori','is_duplicate','yhat','d','q1_set_diff','q2_set_diff']]
    
    """
    
    q1_set_diff = df.apply(lambda x: set_diff(x.q1_wt, x.q2_wt), axis=1)
    q2_set_diff = df.apply(lambda x: set_diff(x.q2_wt, x.q1_wt), axis=1)
    df['q1_set_diff'] = q1_set_diff # for broadcast error
    df['q2_set_diff'] = q2_set_diff
    df['q1_set_diff_len'] = df.q1_set_diff.map(lambda x: len(x))
    df['q2_set_diff_len'] = df.q2_set_diff.map(lambda x: len(x))
    df['q1_count_tagJ'] = df.apply(lambda x: count_tag(x.q1_set_diff, 'J'), axis=1)
    df['q2_count_tagJ'] = df.apply(lambda x: count_tag(x.q2_set_diff, 'J'), axis=1)
    df['q1_count_tagN'] = df.apply(lambda x: count_tag(x.q1_set_diff, 'N'), axis=1)
    df['q2_count_tagN'] = df.apply(lambda x: count_tag(x.q2_set_diff, 'N'), axis=1)
    
    
    df['q1_most_sim'] = df.apply(lambda x: most_sim(x.q1_set_diff, x.q2), axis=1)
    df['q2_most_sim'] = df.apply(lambda x: most_sim(x.q2_set_diff, x.q1), axis=1)
    
    df['q1_most_sim_max'] = df.q1_most_sim.map(lambda x: np.max(x) if len(x)>0 else -1)
    df['q2_most_sim_max'] = df.q2_most_sim.map(lambda x: np.max(x) if len(x)>0 else -1)
    df['q1_most_sim_mean'] = df.q1_most_sim.map(lambda x: np.mean(x) if len(x)>0 else -1)
    df['q2_most_sim_mean'] = df.q2_most_sim.map(lambda x: np.mean(x) if len(x)>0 else -1)
    df['q1_most_sim_min'] = df.q1_most_sim.map(lambda x: np.min(x) if len(x)>0 else -1)
    df['q2_most_sim_min'] = df.q2_most_sim.map(lambda x: np.min(x) if len(x)>0 else -1)
    
    df['q1_weights'] = df.q1_set_diff.map(get_weights)
    df['q2_weights'] = df.q2_set_diff.map(get_weights)
    
    df['q1_weights_max'] = df.q1_weights.map(lambda x: np.max(x) if len(x)>0 else -1)
    df['q2_weights_max'] = df.q2_weights.map(lambda x: np.max(x) if len(x)>0 else -1)
    df['q1_weights_mean'] = df.q1_weights.map(lambda x: np.mean(x) if len(x)>0 else -1)
    df['q2_weights_mean'] = df.q2_weights.map(lambda x: np.mean(x) if len(x)>0 else -1)
    df['q1_weights_min'] = df.q1_weights.map(lambda x: np.min(x) if len(x)>0 else -1)
    df['q2_weights_min'] = df.q2_weights.map(lambda x: np.min(x) if len(x)>0 else -1)
    
    col = df.dtypes[df.dtypes!='object'].index.tolist()
    
    return df[col]
#==============================================================================


train = main(train)
test = main(test)

utils.to_csv(train, test, 'f010')






print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))







