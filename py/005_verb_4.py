# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:58:46 2017

@author: Kazuki

nohup python -u 006_verb.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
import utils
from nltk import stem
pt = stem.PorterStemmer().stem

from gensim.models import Doc2Vec
d2v = Doc2Vec.load('../nlp_source/d2v/enwiki_dbow/doc2vec.bin')
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)


train, test = utils.load(4,1)

stop_verbs = {'do','does','did','done','is','am','are','be','been',
              'being','was','were',"'s","'m"}


"""
nltk.pos_tag(nltk.word_tokenize('How can I be a good geologist ?'))

s='How do I use Twitter as a business source ? '
s='What is the best way to gain confidence ? '
s='How can I find an IT job in Japan ? '
s='How I can speak English fluently ? '
s='Should I switch from Spotify to Apple Music ? '
s='What`s the best way to get rid of porn addiction ? '


"""
#==============================================================================
# def
#==============================================================================

def get_V(wt):
    return [w for w,t in wt if t.startswith('V') and w.lower() not in stop_verbs]

def V_share(vlist1, vlist2):
    vlist2 = list(map(pt, vlist2))
    return sum([1 for v1 in vlist1 if pt(v1) in vlist2])

def V_similarity_d2v(vlist1, vlist2):
    try:
        return d2v.n_similarity(vlist1, vlist2)
    except:
        return -1

def V_similarity_w2v(vlist1, vlist2):
    try:
        return w2v.n_similarity(vlist1, vlist2)
    except:
        return -1

def main(df):
    """
    cv, imp = utils.load_cv()
    cv['d'] = abs(cv.is_duplicate - cv.yhat)
    train = train.merge(cv[['id','yhat','d','q1_ori', 'q2_ori']], on='id')
    
    
    df = train.sample(999)
    
    
    df = df[['q1_ori', 'q2_ori', 'is_duplicate', 'yhat','d', 'q1_V','q2_V']]
    
    """
    df['q1_V'] = df.q1_wt.map(get_V)
    df['q2_V'] = df.q2_wt.map(get_V)
    df['q1_V_len'] = df.q1_V.map(lambda x: len(x))
    df['q2_V_len'] = df.q2_V.map(lambda x: len(x))
    df['V_share'] = df.apply(lambda x: V_share(x.q1_V, x.q2_V), axis=1)
    df['q1_V_ratio'] = df['V_share']/df['q1_V_len']
    df['q2_V_ratio'] = df['V_share']/df['q2_V_len']
    
    # vec
    df['V_similarity_d2v'] = df.apply(lambda x: V_similarity_d2v(x.q1_V, x.q2_V), axis=1)
    df['V_similarity_w2v'] = df.apply(lambda x: V_similarity_w2v(x.q1_V, x.q2_V), axis=1)
    
    col = df.dtypes[df.dtypes!='object'].index.tolist()

    return df[col]


#==============================================================================

train = main(train)
test = main(test)


utils.to_csv(train, test, 'f005')


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))


