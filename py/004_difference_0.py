# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:48:26 2017

@author: Kazuki

nohup python 005_difference.py &


"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
stops = utils.stops
import multiprocessing as mp
total_proc = 2
#from sklearn.metrics import roc_auc_score

w2v_google = utils.load_vec('w2v-google')
w2v_quora = utils.load_vec('w2v-quora-0')

train, test = utils.load(0, 1)

#==============================================================================
# def
#==============================================================================

def after_between(s):
    """
    input str
    return list
    """
    s = s.split()
    ix = s.index('between')
    if '?' in s:
        ques_ix = s.index('?')
    else:
        ques_ix = len(s)
    return s[ix+1:ques_ix]

def different_from(s):
    """
    input str
    return list
    """
    s = s.split()
    ix = s.index('from')
    s = s[:ix-1]+s[ix+1:]
    return [w for w in s if w.lower() not in stops]

def convert(s):
    if 'between' in s:
        return after_between(s)
    elif 'from' in s:
        return different_from(s)
    else:
        s = sum([w.split() for w in utils.noun_chunks(s)], [])
        return s

def diff_w2v_google(s1, s2):
    try:
        if 'differ' not in s1 and 'differ' not in s2:
            return -1
        s1 = utils.valid_words(w2v_google, convert(s1))
        s2 = utils.valid_words(w2v_google, convert(s2))
        return w2v_google.n_similarity(s1, s2)
    except:
        return -1

def diff_w2v_quora(s1, s2):
    try:
        if 'differ' not in s1 and 'differ' not in s2:
            return -1
        s1 = utils.valid_words(w2v_quora, convert(s1))
        s2 = utils.valid_words(w2v_quora, convert(s2))
        return w2v_quora.n_similarity(s1, s2)
    except:
        return -1

def diff_w2v_set_diff(s1, s2):
    try:
        if 'differ' not in s1 and 'differ' not in s2:
            return -1
        s1 = set(utils.valid_words(w2v_google, convert(s1)))
        s2 = set(utils.valid_words(w2v_google, convert(s2)))
        return w2v_google.n_similarity(s1-s2, s2-s1)
    except:
        return -1

def len_valid(s1, s2):
    try:
        if 'differ' not in s1 and 'differ' not in s2:
            return -1
        s1 = set(utils.valid_words(w2v_google, convert(s1)))
        s2 = set(utils.valid_words(w2v_google, convert(s2)))
        return (len(s1)==len(s2))*1
    except:
        return -1

def important_noun_after_between(s):
    nouns = []
    noun = ''
    for w in after_between(s):
        if w in ['the','a','an']:
            continue
        if w in [',','and','&']:
            nouns.append(noun.strip())
            noun = ''
            continue
        elif w in ['?','in']:
            break
        else:
            noun += w+' '
    nouns.append(noun.strip())
    return nouns


def isin_important_noun(s1,s2):
    # double between
    if ' between ' in s1 and ' between ' in s2:
        s1_ = important_noun_after_between(s1)
        s2_ = important_noun_after_between(s2)
        return (len(set(s1_) - set(s2_))==0 and len(set(s2_) - set(s1_))==0)*1
    elif ' between ' in s1:
        s1 = important_noun_after_between(s1)
        cnt=0
        for w1 in s1:
            if w1 in s2:
                cnt+=1
        return (cnt==len(s1))*1
            
    elif ' between ' in s2:
        s2 = important_noun_after_between(s2)
        cnt=0
        for w2 in s2:
            if w2 in s1:
                cnt+=1
        return (cnt==len(s2))*1
    return -1

def count_noun(s1, s2):
    try:
        s1 = utils.valid_words(w2v_google, convert(s1))
        s2 = utils.valid_words(w2v_google, convert(s2))
        return len(s1)==len(s2)*1
    except:
        return -1

def differ_index(s):
    try:
        return s.index('differ')
    except:
        return -1

def isin_differ(s):
    if 'difference' in s and 'between' in s:
        return 1
    elif ' differ' in s and 'from ' in s:
        return 1
    return 0
    
def main(p):
    """
    w='differ'
    df = train[train.q1.str.contains(w)|train.q2.str.contains(w)].sample(9999)
    
    df['diff_w2v'] = df.apply(lambda x: diff_w2v(x['q1'], x['q2']),axis=1)
    df['diff_w2v_set_diff'] = df.apply(lambda x: diff_w2v_set_diff(x['q1'], x['q2']),axis=1)
    df['len_valid'] = df.apply(lambda x: len_valid(x['q1'], x['q2']),axis=1)
    df['isin_important_noun'] = df.apply(lambda x: isin_important_noun(x['q1'], x['q2']),axis=1)
    
    roc_auc_score(df.is_duplicate, df.diff_w2v)
    """
    
    if p==0:
        df = train
    elif p==1:
        df = test
    
    df['diff_w2v_google']   = df.apply(lambda x: diff_w2v_google(x['q1'], x['q2']),axis=1)
    df['diff_w2v_quora']    = df.apply(lambda x: diff_w2v_quora(x['q1'], x['q2']),axis=1)
    df['diff_w2v_set_diff'] = df.apply(lambda x: diff_w2v_set_diff(x['q1'], x['q2']),axis=1)
    df['len_valid'] = df.apply(lambda x: len_valid(x['q1'], x['q2']),axis=1)
    df['isin_important_noun'] = df.apply(lambda x: isin_important_noun(x['q1'], x['q2']),axis=1)
    df['diff_count_noun'] = df.apply(lambda x: count_noun(x['q1'], x['q2']),axis=1)
    df['q1_differ_index'] = df.q1.map(differ_index)
    df['q2_differ_index'] = df.q2.map(differ_index)
    df['isin_differ'] = df.q1.map(isin_differ) + df.q2.map(isin_differ)
    
    if 'is_duplicate' in df.columns:
        df.drop('is_duplicate', axis=1, inplace=1)
    col = df.dtypes[df.dtypes!='object'].index.tolist()
    
    path = '~/Quora/feature/'
    name = 'f004'
    if 'id' in df.columns:
        df[col].to_csv(path+'train_{0}.csv.gz'.format(name), index=False, compression='gzip')
    else:
        df[col].to_csv(path+'test_{0}.csv.gz'.format(name), index=False, compression='gzip')
    
    return

#==============================================================================
# main
#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))



