# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:04:39 2017

@author: Kazuki
"""

import pandas as pd
import re
import numpy as np
from nltk import stem
pt = stem.PorterStemmer().stem
from nltk.corpus import stopwords
stops = set(stopwords.words("english")+['could','would','might'])
kigou = set([':', ';', '.', ',', '?', '/', '(', ')', '!', '-', '"'])
# Import spacy and English models
import spacy
nlp = spacy.load('en')
from glob import glob

#==============================================================================
# def
#==============================================================================
def preprocessing(txt):
    try:
        txt = txt.replace('?',' ? ')
        txt = txt.replace("What's", 'What is')
        txt = txt.replace(" U.S. ", ' US ')
        txt = txt.replace('\u201c','').replace('\u201d','')\
                .replace('\u2014'," ").replace('\u2018',"")\
                .replace('\u2019',"").replace('\u2026',".")\
                .replace('(',' ( ').replace(')',' ) ')\
                .replace('!',' ! ')\
                .replace('"','').replace('/',' / ')
        txt = re.sub(r"(\\n|\\r)", r"", txt)
        # insert space in kigou
        txt = re.sub(r"([a-z]|[0-9]|[A-Z]+)(\?|\!|\,|\:|\;|\/|\+|\*)([a-z]|[0-9]|\s|[A-Z]+)", r"\1 \2 \3", txt)
        txt = re.sub(r"([a-z]|[0-9]|\s|[A-Z]+)(\?|\!|\,|\:|\;|\/|\+|\*)", r"\1 \2 ", txt)
        txt = re.sub(r"([a-z]|[A-Z])(\.)(\s)", r"\1 \2\3", txt)
        txt = re.sub(r"([a-z]|[A-Z])(\.)($)", r"\1 \2\3", txt)
        txt = re.sub(r"(\s)(\')([a-z]|[0-9]|[A-Z])", r"\1 \2 \3", txt)
        txt = re.sub(r"([a-z]|[0-9]|[A-Z])(\')(\s)", r"\1 \2 \3", txt)
        txt = re.sub(r"([0-9])(000)(\s|$)", r"\1k", txt)
        txt = txt.replace('c + +','c++')
        txt = txt.replace('C + +','c++')
        txt = txt.replace('-year-old',' year old')
        txt = re.sub(r"([0-9])(kg)(\s|$)", r"\1 \2 \3", txt)
        txt = re.sub(r"([0-9])(km)(\s|$)", r"\1 \2 \3", txt)
        txt = re.sub(r"([0-9])(kgs)(\s|$)", r"\1 \2 \3", txt)
        txt = re.sub(r"([0-9])(gb)(\s|$)", r"\1 \2 \3", txt)
        txt = txt.replace('  ', ' ') # unecessary??
        return txt.strip()
    
    except:
        print(txt)
        return txt

def find_noun(wt):
    try:
        # search nearest noun(Objective)
        n = [w for w,t in wt if 'NN' in t][0]
        return n
    except:
        return
        
def find_noun_after_prep(prep, s, is_all=True):
    """
    prep = 'in'
    
    s = 'What is the step by step guide to invest in share market in india ?'
    return [market, india]
    
    s = 'How hard is it to fake appearing spiritual in Christian community ?'
    return [community]
    
    prep = 'on'
    s ='How do you manage inventory on Amazon ?'
    return [Amazon]
    
    """
    
    try:
        if prep not in s.split():
            return
        
        ret = []
        wt = get_tag(s)
            
        sw_in = False
        noun = None
        
        for w,t in wt:
            
            if sw_in == False and prep == w:
                sw_in = True
                continue
            if sw_in:
                if t.startswith('NN'):
                    noun = w
                    
                if noun is not None and t != 'NN': # NN -> ??
                    ret.append(noun)
                    sw_in = False
                    noun = None
                    if prep == w:
                        sw_in = True
                        continue
            else:
                continue
        if is_all:
            return ret
        else:
            return ret[0]
    except:
        if is_all:
            return []
        else:
            return


def match(w1, w2):
    if w1 is None and w2 is None:
        return '0'
    elif w1 is None or w2 is None:
        return '1'
    elif w1 == w2:
        return 'p'
    elif pt(w1) == pt(w2):
        return 's'
    # TODO: similarity
    else:
        return 'N'

def list_match(li1, li2):
    
    if len(li1) == len(li2) == 0:
        return '0'
    elif len(li1)==0 or len(li2) == 0:
        return '1'
    elif li1 == li2:
        return 'p'
        
    li1_lo = [l.lower() for l in li1]
    li2_lo = [l.lower() for l in li2]
    if li1_lo == li2_lo:
        return 'p-lo'
        
    li1_st = [pt(l) for l in li1_lo]
    li2_st = [pt(l) for l in li2_lo]
    if li1_st == li2_st:
        return 'p-stem'
    for i in li1:
        if i in li2:
            return 'in'
    return 'N'
    
def get_tag(s):
    s = nlp(s)
    return [(token.text, token.tag_) for token in s]
    
def get_noun(s):
    s = nlp(s)
    return [token.text for token in s if token.tag_.startswith('N')]

def noun_chunks(s):
    s = nlp(s)
    chunks = [chunk.text for chunk in s.noun_chunks]
    ret = []
    for chunk in chunks:
        chunk = ' '.join([c for c in chunk.split() if c.lower() not in stops])
        if len(chunk)>0:
            ret.append(chunk) # skip ['']
    return ret

def valid_words(model, words):
    return [w for w in words if w in model]

def d2v_similarity(s1list, s2list, source='wiki'):
    
    if len(s1list) == len(s2list) == 0:
        return 0
    
    try:
        if source=='wiki':
            model = d2v_wiki
            
        elif source=='apnews':
            model = d2v_apnews
            
        elif source=='quora':
            model = d2v_quora
            
        else:
            raise Exception('unknon source:', source)
        
        s1list = valid_words(model, s1list)
        s2list = valid_words(model, s2list)
        return model.n_similarity(s1list, s2list)
        
    except:
        return -1
    
def w2v_similarity(s1list, s2list, source='google'):
    
    if len(s1list) == len(s2list) == 0:
        return 0
    
    try:
        if source=='google':
            model = w2v_google
        
        elif source=='wiki':
            model = w2v_wiki
        
        elif source=='apnews':
            model = w2v_apnews
        
        elif source=='quora':
            model = w2v_quora
            
        else:
            raise Exception('unknoun source:', source)
            
        s1list = valid_words(model, s1list)
        s2list = valid_words(model, s2list)
        return model.n_similarity(s1list, s2list)
        
    except:
        return -1


def glv_similarity(s1list, s2list, source='840'):
    
    if len(s1list) == len(s2list) == 0:
        return 0
    
    try:
        if source=='840':
            model = glv_common840
            
        elif source=='twitter':
            model = glv_twitter27
            
        else:
            raise Exception('unknon source:', source)
            
        s1list = valid_words(model, s1list)
        s2list = valid_words(model, s2list)
        return model.n_similarity(s1list, s2list)
    except:
        return -1

def sen2vec(model, words, default=300):
    try:
        words = valid_words(model, words)
        if len(words)==0:
            raise
        vec   = [model[w] for w in words]
        return np.array(vec).mean(axis=0)
    except:
        return np.ones(default)*-1


def load_all_vec():
    global d2v_wiki, d2v_apnews, d2v_quora
    global w2v_google, w2v_wiki, w2v_apnews, w2v_quora
    global glv_common840, glv_twitter27
    
    print('loading doc2vec')
    from gensim.models import Doc2Vec
    d2v_wiki = Doc2Vec.load('../nlp_source/d2v/enwiki_dbow/doc2vec.bin')
    d2v_apnews = Doc2Vec.load('../nlp_source/d2v/apnews_dbow/doc2vec.bin')
    d2v_quora = Doc2Vec.load('../nlp_source/d2v/Quora/Quora-doc2vec-D100-min_count3.model')
    
    print('loading word2vec')
    from gensim.models import KeyedVectors
    w2v_google = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)
    
    from gensim.models import Word2Vec
    w2v_wiki = Word2Vec.load('../nlp_source/w2v/wiki_sg/word2vec.bin')
    w2v_apnews = Word2Vec.load('../nlp_source/w2v/apnews_sg/word2vec.bin')
    w2v_quora = Word2Vec.load('../nlp_source/w2v/Quora/Quora-word2vec-D100-min_count3.model')
    
    print('loading GloVe')
    glv_common840 = KeyedVectors.load_word2vec_format('../nlp_source/w2v/glove.840B.300d.gz')
    glv_twitter27 = KeyedVectors.load_word2vec_format('../nlp_source/w2v/glove.twitter.27B.200d.gz')

def load_vec(name):
    
    from gensim.models import Doc2Vec
    from gensim.models import KeyedVectors
    from gensim.models import Word2Vec
    
    if name=='d2v-wiki':
        print('loding:{}'.format(name))
        return Doc2Vec.load('../nlp_source/d2v/enwiki_dbow/doc2vec.bin')
    
    elif name=='d2v-apnews':
        print('loding:{}'.format(name))
        return Doc2Vec.load('../nlp_source/d2v/apnews_dbow/doc2vec.bin')
    
    elif name=='d2v-quora-0':
        print('loding:{}'.format(name))
        return Doc2Vec.load('../nlp_source/d2v/Quora/Quora-doc2vec-0-D100-min_count3.model')
    
    elif name=='w2v-google':
        print('loding:{}'.format(name))
        return KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)
    
    elif name=='w2v-wiki':
        print('loding:{}'.format(name))
        return Word2Vec.load('../nlp_source/w2v/wiki_sg/word2vec.bin')
        
    elif name=='w2v-apnews':
        print('loding:{}'.format(name))
        return Word2Vec.load('../nlp_source/w2v/apnews_sg/word2vec.bin')
        
    elif name=='w2v-quora-0':
        print('loding:{}'.format(name))
        return Word2Vec.load('../nlp_source/w2v/Quora/Quora-word2vec-0-D100-min_count3.model')
    
    elif name=='w2v-quora-2':
        print('loding:{}'.format(name))
        return Word2Vec.load('../nlp_source/w2v/Quora/Quora-word2vec-2-D100-min_count3.model')
    
    
    
    
    
    else:
        raise Exception('Unknown name:', name)



def merge_y(train):
    y_train = pd.read_csv('../input/train.csv.zip', usecols=['id','is_duplicate'])
    train = pd.merge(train, y_train, on='id', how='left')
    return train

def load(n, debug=False):
    """
    n = 0~2
    """
    if n==0:
        print('load prep')
        try:
            train = pd.read_csv('../input/mk/train_prep.csv.gz')
            test  = pd.read_csv('../input/mk/test_prep.csv.gz')
        except:
            train = pd.read_csv('~/Quora/input/mk/train_prep.csv.gz')
            test  = pd.read_csv('~/Quora/input/mk/test_prep.csv.gz')
        
    elif n==1:
        print('load stem')
        try:
            train = pd.read_csv('../input/mk/train_stem.csv.gz')
            test  = pd.read_csv('../input/mk/test_stem.csv.gz')
        except:
            train = pd.read_csv('~/Quora/input/mk/train_stem.csv.gz')
            test  = pd.read_csv('~/Quora/input/mk/test_stem.csv.gz')
        
    elif n==2:
        print('load stop')
        try:
            train = pd.read_csv('../input/mk/train_stop.csv.gz')
            test  = pd.read_csv('../input/mk/test_stop.csv.gz')
        except:
            train = pd.read_csv('~/Quora/input/mk/train_stop.csv.gz')
            test  = pd.read_csv('~/Quora/input/mk/test_stop.csv.gz')
        
    elif n==3:
        print('load stop-stem')
        try:
            train = pd.read_csv('../input/mk/train_stop-stem.csv.gz')
            test  = pd.read_csv('../input/mk/test_stop-stem.csv.gz')
        except:
            train = pd.read_csv('~/Quora/input/mk/train_stop-stem.csv.gz')
            test  = pd.read_csv('~/Quora/input/mk/test_stop-stem.csv.gz')
    
        
    elif n==4:
        print('load tag')
        try:
            train = pd.read_pickle('../input/mk/train_tag.p')
            test  = pd.read_pickle('../input/mk/test_tag.p')
        except:
            train = pd.read_pickle('~/Quora/input/mk/train_tag.p')
            test  = pd.read_pickle('~/Quora/input/mk/test_tag.p')
        
    elif n==5:
        print('load ent')
        try:
            train = pd.read_pickle('../input/mk/train_ent.p')
            test  = pd.read_pickle('../input/mk/test_ent.p')
        except:
            train = pd.read_pickle('~/Quora/input/mk/train_ent.p')
            test  = pd.read_pickle('~/Quora/input/mk/test_ent.p')
            
    elif n==6:
        print('load chunks')
        try:
            train = pd.read_pickle('../input/mk/train_chunks.p')
            test  = pd.read_pickle('../input/mk/test_chunks.p')
        except:
            train = pd.read_pickle('~/Quora/input/mk/train_chunks.p')
            test  = pd.read_pickle('~/Quora/input/mk/test_chunks.p')
    
        
    elif n==7:
        print('load tag-stop')
        try:
            train = pd.read_pickle('../input/mk/train_tag-stop.p')
            test  = pd.read_pickle('../input/mk/test_tag-stop.p')
        except:
            train = pd.read_pickle('~/Quora/input/mk/train_tag-stop.p')
            test  = pd.read_pickle('~/Quora/input/mk/test_tag-stop.p')
            
    elif n==8:
        print('load and-or_0')
        try:
            train = pd.read_pickle('../input/mk/train_and-or_0.p')
            test  = pd.read_pickle('../input/mk/test_and-or_0.p')
        except:
            train = pd.read_pickle('~/Quora/input/mk/train_and-or_0.p')
            test  = pd.read_pickle('~/Quora/input/mk/test_and-or_0.p')
            
    elif n==9:
        print('load and-or_2')
        try:
            train = pd.read_pickle('../input/mk/train_and-or_2.p')
            test  = pd.read_pickle('../input/mk/test_and-or_2.p')
        except:
            train = pd.read_pickle('~/Quora/input/mk/train_and-or_2.p')
            test  = pd.read_pickle('~/Quora/input/mk/test_and-or_2.p')
            
    elif n==10:
        print('load syn_0')
        try:
            train = pd.read_csv('../input/mk/train_syn_0.csv.gz')
            test  = pd.read_csv('../input/mk/test_syn_0.csv.gz')
        except:
            train = pd.read_csv('~/Quora/input/mk/train_syn_0.csv.gz')
            test  = pd.read_csv('~/Quora/input/mk/test_syn_0.csv.gz')
            
    elif n==11:
        print('load syn_2')
        try:
            train = pd.read_csv('../input/mk/train_syn_2.csv.gz')
            test  = pd.read_csv('../input/mk/test_syn_2.csv.gz')
        except:
            train = pd.read_csv('~/Quora/input/mk/train_syn_2.csv.gz')
            test  = pd.read_csv('~/Quora/input/mk/test_syn_2.csv.gz')
            
    elif n==15:
        print('load JNV')
        try:
            train = pd.read_pickle('../input/mk/train_JNV.p')
            test  = pd.read_pickle('../input/mk/test_JNV.p')
        except:
            train = pd.read_pickle('~/Quora/input/mk/train_JNV.p')
            test  = pd.read_pickle('~/Quora/input/mk/test_JNV.p')
    
            
    else:
        raise Exception('n 0~8')
    
    if debug:
        train = merge_y(train)
        
    train.dropna(inplace=True)
    test.dropna(inplace=True)
#    test = test.rename(columns={'test_id':'id'})
    
    return train, test

def load_cv():
    
    train = pd.read_csv('../input/train.csv.zip')\
    .rename(columns={'question1':'q1_ori','question2':'q2_ori'})
    
    train = pd.merge(train, pd.read_csv('~/Dropbox/Python/playground/Quora/output/cv.csv.gz',usecols=['id','yhat']),
                     on='id', how='left')
    
    train['d'] = abs(train.is_duplicate - train.yhat)
    train.sort_values('d', ascending=0, inplace=1)
    
#    train = pd.merge(train, pd.read_csv('../input/mk/train_stem.csv.gz')\
#    .rename(columns={'q1':'q1_stem','q2':'q2_stem'}),
#                     on='id', how='left')
#    
#    train = pd.merge(train, pd.read_csv('../input/mk/train_stop.csv.gz')\
#    .rename(columns={'q1':'q1_stop','q2':'q2_stop'}),
#                     on='id', how='left')
    
    imp = pd.read_csv('~/Dropbox/Python/playground/Quora/output/imp.csv.gz')
    
    return train.drop(['qid1', 'qid2'],axis=1), imp
    
def load_tf():
    df = pd.read_csv('../nlp_source/tf.csv.gz')
    keys = df.word.tolist()
    values = df.weight.tolist()
    tf = {k:v for k,v in zip(keys,values)}
    return tf


def load_train(file_in=None, file_remove=None):
    """
    
    
    """
    train = pd.read_csv('../input/train.csv.zip')
    
    if file_in is None and file_remove is None:
        files_ = sorted(glob('../feature/train_f*'))
        
    elif file_remove is None:
        files = sorted(glob('../feature/train_f*'))
        files_ = []
        for f in files:
            for f_ in file_in:
                if f_ in f:
                    files_.append(f)
    elif file_in is None:
        files = sorted(glob('../feature/train_f*'))
        files_ = []
        for f in files:
            if sum([1 for fr in file_remove if fr in f]) == 0:
                files_.append(f)
    else:
        raise
    files_ = sorted(set(files_))
    for f in files_:
        print('loading:{}'.format(f))
        train = pd.merge(train, pd.read_csv(f), on='id', how='left')
        print('shape:{}'.format(train.shape))
    
    return train
    
def load_test(file_in=None, file_remove=None):
    test = pd.read_csv('../input/test.csv.zip')
    
    if file_in is None and file_remove is None:
        files_ = sorted(glob('../feature/test_f*'))
        
    elif file_remove is None:
        files = sorted(glob('../feature/test_f*'))
        files_ = []
        for f in files:
            for f_ in file_in:
                if f_ in f:
                    files_.append(f)
    elif file_in is None:
        files = sorted(glob('../feature/test_f*'))
        files_ = []
        for f in files:
            if sum([1 for fr in file_remove if fr in f]) == 0:
                files_.append(f)
    else:
        raise
    files_ = sorted(set(files_))
    for f in files_:
        print('loading:{}'.format(f))
        test = pd.merge(test, pd.read_csv(f).rename(columns={'id':'test_id'}), 
                        on='test_id', how='left')
        print('shape:{}'.format(test.shape))
    
    return test

def load_009(flist, is_train=True):
    """
    flist: list
    like [0, 1, 2]
    """
    
    if is_train:
        files = ['../feature/train_f009-word-{0}.csv.gz'.format(p) for p in flist]
    else:
        files  = ['../feature/test_f009-word-{0}.csv.gz'.format(p) for p in flist]
    
    for i,f in enumerate(files):
        print('loading...',f)
        _ = pd.read_csv(f)
        if i==0:
            df = _
        else:
            if is_train:
                df = pd.merge(df, _, on='id', how='left')
            else:
                df = pd.merge(df, _, on='test_id', how='left')
    return df

def to_csv(train, test, name):
#    path = '../feature/'
    path = '~/Quora/feature/'
    col = ['q1', 'q2', 'is_duplicate']
    for c in col:
        if c in train.columns:
            train.drop(c, axis=1, inplace=True)
    train.to_csv(path+'train_{0}.csv.gz'.format(name), index=False, compression='gzip')
    
    col = [ 'q1', 'q2']
    for c in col:
        if c in test.columns:
            test.drop(c, axis=1, inplace=True)
    test.to_csv(path+'/test_{0}.csv.gz'.format(name), index=False, compression='gzip')

def down_sampling(X, y, p=0.125):
    
    size = int(X.shape[0]*p)
    pos = X[y==1].sample(size)
    neg = X[y==0]
    #pd.concat([pos,neg], ignore_index=True).is_duplicate.mean()
    X = pd.concat([pos,neg], ignore_index=True)
    y_ = (np.zeros(len(pos)) + 1).tolist() + np.zeros(len(neg)).tolist()
    return X, pd.Series(y_)

def up_sampling(X, y):
    pos_train = X[y == 1]
    neg_train = X[y == 0]
    
    # Now we oversample the negative class
    # There is likely a much more elegant way to do this...
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -=1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print(len(pos_train) / (len(pos_train) + len(neg_train)))
    
    X  = pd.concat([pos_train, neg_train])
    y_ = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train
    
    return X, pd.Series(y_)



def mirroring(X, y):
    """
    q1 -> q3
    q2 -> q1
    q3 -> q2
    
    """
    
    col_q1 = [c for c in X.columns if 'q1' in c]
    col_q2 = [c.replace('q1','q2') for c in col_q1]
    col_q3 = [c.replace('q1','q3') for c in col_q1]
    
    di_1to3 = dict(zip(col_q1,col_q3))
    di_2to1 = dict(zip(col_q2,col_q1))
    di_3to2 = dict(zip(col_q3,col_q2))
    
    X_ = X.rename(columns=di_1to3)
    X_ = X_.rename(columns=di_2to1)
    X_ = X_.rename(columns=di_3to2)
    
    X = pd.concat([X,X_], ignore_index=1)
    y = pd.concat([y,y], ignore_index=1)
    
    return X, y
    
def q1_to_q2(X):
    """
    q1 -> q3
    q2 -> q1
    q3 -> q2
    
    """
    
    col_q1 = [c for c in X.columns if 'q1' in c]
    col_q2 = [c.replace('q1','q2') for c in col_q1]
    col_q3 = [c.replace('q1','q3') for c in col_q1]
    
    di_1to3 = dict(zip(col_q1,col_q3))
    di_2to1 = dict(zip(col_q2,col_q1))
    di_3to2 = dict(zip(col_q3,col_q2))
    
    X_ = X.rename(columns=di_1to3)
    X_ = X_.rename(columns=di_2to1)
    X_ = X_.rename(columns=di_3to2)
        
    return X_
















