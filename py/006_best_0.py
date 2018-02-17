# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:25:12 2017

@author: konodera

nohup python 007_best.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
import utils
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)

train, test = utils.load(0, 1)

#==============================================================================
# def
#==============================================================================
def find_best_noun(s):
    ix_best = s.index('best')
    wt = utils.get_tag(s[ix_best:])
    
    if ' to ' in s:
        ix_TO = min([i for i,(w,t) in enumerate(wt) if t=='TO'])
    else:
        ix_TO = len(wt)
        
    ix_INs = [i for i,(w,t) in enumerate(wt) if t=='IN']
    if len(ix_INs) != 0:
        ix_IN = min(ix_INs)
    else:
        ix_IN = len(wt)
    
    if ix_TO != ix_IN:
        ix_min = min(ix_TO, ix_IN)
        return wt[ix_min-1][0]
        
    sw_NN = False
    for w,t in wt:
        if 'NN' in t:
            sw_NN = True
        if sw_NN:
            if 'NN' in t:
                noun = w
            else:
                return noun
        else:
            continue
    return


def best(s1, s2):
    """
    match: noun, condition, 
    
    
    
    """
    if (' way ' in s1 or ' ways ' in s1) or (' way ' in s2 or ' ways ' in s2):
        return -3
    try:
        # double best
        if ' best ' in s1 and ' best ' in s2:
            s1_ = s1.split(); s2_ = s2.split()
            
            if s1_[0] in ['What','Which'] and s2_[0] in ['What','Which'] and\
            s1_[1] in ['is','are'] and s2_[1] in ['is','are'] and\
            s1_[2:4] == s2_[2:4] == ['the','best']:
                noun1 = utils.find_noun_after_prep('best', s1, 0)
                noun2 = utils.find_noun_after_prep('best', s2, 0)
                return w2v.similarity(noun1, noun2)
                
            return 0
        # only s1 or s2
        elif ('best' in s1 and 'best' not in s2) or ('best' not in s1 and 'best' in s2):
            pass # TODO: 
                    
        return -1
    except:
        return -2

"""
tmp = train[train.q1.str.contains('best') | train.q1.str.contains('best')]
tmp = test[test.question1.str.contains('best') | test.question2.str.contains('best')]


tmp['tmp'] = tmp.apply(lambda x: best(x.q1, x.q2), axis=1)
ct = pd.crosstab(tmp['tmp'],tmp['is_duplicate'],normalize='index').sort_values(1,ascending=False)
ct.columns = ['p0','p1']
ct = pd.concat([ct,pd.crosstab(tmp['tmp'],tmp['is_duplicate'])],axis=1)

raise

"""
#==============================================================================

train['best'] = train.apply(lambda x: best(x['q1'], x['q2']),axis=1)
test['best']  = test.apply(lambda x: best(x['q1'], x['q2']),axis=1)


utils.to_csv(train, test, 'f006')


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))






