# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:24:13 2017

@author: konodera

https://www.kaggle.com/the1owl/quora-question-pairs/matching-que-for-quora-end-to-end-0-33719-pb/notebook

nohup python 102_from_the1owl.py &

"""

import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))
#0.33719 on Public Leaderboard

import nltk
import utils
stops = utils.stops
import difflib


#==============================================================================
# def
#==============================================================================
def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def get_unigrams(que):
    return [word for word in nltk.word_tokenize(que.lower()) if word not in stops]

def get_common_unigrams(row):
    return len( set(row["unigrams_q1"]).intersection(set(row["unigrams_q2"])) )

def get_common_unigram_ratio(row):
    return float(row["unigrams_common_count_from_the1owl"]) / max(len( set(row["unigrams_q1"]).union(set(row["unigrams_q2"])) ),1)

def get_bigrams(que):
    return [i for i in nltk.ngrams(que, 2)]

def get_common_bigrams(row):
    return len( set(row["bigrams_q1"]).intersection(set(row["bigrams_q2"])) )

def get_common_bigram_ratio(row):
    return float(row["unigrams_common_count_from_the1owl"]) / max(len( set(row["bigrams_q1"]).union(set(row["bigrams_q2"])) ),1)

def main(df, suf):
    
    df['q1_nouns'] = df['q1'].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    df['q2_nouns'] = df['q2'].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    
#    df['z_len1'] = df['q1'].map(lambda x: len(str(x)))
#    df['z_len2'] = df['q2'].map(lambda x: len(str(x)))
#    df['z_word_len1'] = df['q1'].map(lambda x: len(str(x).split()))
#    df['z_word_len2'] = df['q2'].map(lambda x: len(str(x).split()))
    df['noun_match_from_the1owl'] = df.apply(lambda r: sum([1 for w in r['q1_nouns'] if w in r['q2_nouns']]), axis=1)
    
    df['match_ratio_from_the1owl'] = df.apply(lambda r: diff_ratios(r['q1'], r['q2']), axis=1)
    
    df["unigrams_q1"] = df['q1'].apply(lambda x: get_unigrams(str(x)))
    df["unigrams_q2"] = df['q2'].apply(lambda x: get_unigrams(str(x)))
    df["unigrams_common_count_from_the1owl"] = df.apply(lambda r: get_common_unigrams(r),axis=1)
    df["unigrams_common_ratio_from_the1owl"] = df.apply(lambda r: get_common_unigram_ratio(r), axis=1)
    df["bigrams_q1"] = df["unigrams_q1"].apply(lambda x: get_bigrams(x))
    df["bigrams_q2"] = df["unigrams_q2"].apply(lambda x: get_bigrams(x)) 
    df["bigrams_common_count_from_the1owl"] = df.apply(lambda r: get_common_bigrams(r),axis=1)
    df["bigrams_common_ratio_from_the1owl"] = df.apply(lambda r: get_common_bigram_ratio(r), axis=1)
    
    di = {c:c+suf for c in df.columns if '_from_the1owl' in c}
    df = df.rename(columns=di)
    
    col = [c for c in df.columns if '_from_the1owl' in c or 'id' in c]
    
    
    return df[col]



#==============================================================================

train, test = utils.load(0)
train = main(train, '').fillna(-1)
test  = main(test, '').fillna(-1)
utils.to_csv(train, test, 'f102-0')

train, test = utils.load(1)
train = main(train, '-stem').fillna(-1)
test  = main(test, '-stem').fillna(-1)
utils.to_csv(train, test, 'f102-1')

train, test = utils.load(2)
train = main(train, '-stop').fillna(-1)
test  = main(test, '-stop').fillna(-1)
utils.to_csv(train, test, 'f102-2')

train, test = utils.load(3)
train = main(train, '-stst').fillna(-1)
test  = main(test, '-stst').fillna(-1)
utils.to_csv(train, test, 'f102-3')

#train, test = utils.load(9)
#train = main(train, '-syn1').fillna(-1)
#test  = main(test, '-syn1').fillna(-1)
#utils.to_csv(train, test, 'f102-9')
















print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))



