# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 07:33:01 2017

@author: konodera
"""

import pandas as pd
from itertools import product
from collections import defaultdict
import utils
stops = utils.stops
w2v_google = utils.load_vec('w2v-google')
w2v_quora = utils.load_vec('w2v-quora-0')

# param
sim_value = 0.5


train = pd.merge(pd.read_pickle('../input/mk/train_and-or_0.p'),
                 pd.read_csv('../input/train.csv.zip', usecols=['id','is_duplicate']),
                 on='id', how='left')

train_1 = train.ix[(train.is_duplicate==1) & (train.q1_set_diff.map(lambda x: len(x))==1) & (train.q2_set_diff.map(lambda x: len(x))==1)]


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

train_1['set_diff_most-sim'] = train_1.apply(lambda x: most_sim(x.q1_set_diff, x.q2_set_diff), axis=1)


"""

train['set_diff_most-sim'] = train.apply(lambda x: most_sim(x.q1_set_diff, x.q2_set_diff), axis=1)

train_ = train[train['set_diff_most-sim'].map(len)>0]
train_ = train_[train_['set_diff'].map(len)<6]

import os
os.system("say 'done'")
from collections import Counter

synonyms = sum(train.ix[train.is_duplicate==1]['set_diff_most-sim'].tolist(),[])
synonyms = ['<->'.join(x) for x in synonyms]
synonyms = Counter(synonyms)



antonyms = sum(train.ix[train.is_duplicate==0]['set_diff_most-sim'].tolist(),[])
antonyms = ['<->'.join(x) for x in antonyms]
antonyms = Counter(antonyms)

df = pd.DataFrame(list(synonyms.items()),
                  columns=['word', 'cnt_1']).set_index('word')

df_ = pd.DataFrame(list(antonyms.items()),
                  columns=['word', 'cnt_0']).set_index('word')

df = pd.concat([df,df_], axis=1)








train_0 = train.ix[(train.is_duplicate==0) & (train.q1_set_diff.map(lambda x: len(x))==1) & (train.q2_set_diff.map(lambda x: len(x))==1)]

train_0['set_diff_most-sim'] = train_0.apply(lambda x: most_sim(x.q1_set_diff, x.q2_set_diff), axis=1)

train_ = pd.concat([train_1, train_0])

test_ = test.ix[ (test.q1_set_diff.map(lambda x: len(x))==1) & (test.q2_set_diff.map(lambda x: len(x))==1)]



"""

synonym_set = sum(train_1['set_diff_most-sim'].tolist(),[])

synonym = defaultdict(str)

for w1, w2 in synonym_set:
    synonym[w1] += w2 + ' '
    synonym[w2] += w1 + ' '


for k,v in synonym.items():
#    print(k,v)
    synonym[k] = ' '.join(set(v.split()))


synonym = pd.DataFrame(list(synonym.items()),
                  columns=['k','v'])

synonym.v = synonym.v.map(lambda x: x.split())

synonym.to_pickle('../nlp_source/synonym_1on1_S{}_0.p'.format(sim_value))
#synonym = pd.read_pickle('../nlp_source/synonym_1on1_S{}_2.p'.format(sim_value))


syn_k = synonym.k.tolist()
syn_v = synonym.v.tolist()

synonym = defaultdict(list)
for k,v in zip(syn_k, syn_v):
    synonym[k] += v

#==============================================================================
# apply
#==============================================================================
train, test = utils.load(2, 1)

def set_diff(s1, s2):
    s2 = set(s2.split())
    return [w1 for w1 in s1.split() if w1 not in s2]

def get_synonym(s1, s2):
    """
    return:
    [(w1, syn), (w1, syn)]
    """
    ret = []
    for w1 in s1:
        if w1 in synonym:
            syns = synonym.get(w1)
            for syn in syns:
                if syn in s2:
                    s2.remove(syn)
                    # w1 -> syn
                    ret.append([w1, syn])
    return ret

def to_synonym(s1, syns):
    if len(syns)==0:
        return s1
    s1 = s1.split()
    s1_ret = s1[:]
    old = [syn[0] for syn in syns]
    new = [syn[1] for syn in syns]
    for i,w1 in enumerate(s1):
        if w1 in old:
            ix = old.index(w1)
            s1_ret[i] = new.pop(ix)
            old.pop(ix)
            if len(old)==0:
                break
    return ' '.join(s1_ret)

def main(p):
    """
    df = train.sample(999)
    """
    if p==0:
        df = train
    elif p==1:
        df = test
    
    df['q1_set_diff'] = df.apply(lambda x: set_diff(x.q1, x.q2), axis=1)
    df['q2_set_diff'] = df.apply(lambda x: set_diff(x.q2, x.q1), axis=1)
    df['synonyms'] = df.apply(lambda x: get_synonym(x.q1_set_diff, x.q2_set_diff), axis=1)
    df['q1'] = df.apply(lambda x: to_synonym(x.q1, x.synonyms), axis=1)
    
    if p==0:
        df[['id','q1','q2']].to_csv('../input/mk/train_syn_0.csv.gz', index=0, compression='gzip')
        print('finish train_syn1')
        
    elif p==1:
        df[['test_id','q1','q2']].to_csv('../input/mk/test_syn_0.csv.gz', index=0, compression='gzip')
        print('finish test_syn1')
        
    return



import multiprocessing as mp
total_proc = 2
pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))







print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

