# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 07:31:33 2017

@author: Kazuki

nohup ipython 000-1_spacy.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

# Import spacy and English models
import spacy
nlp = spacy.load('en')
import utils
stops = utils.stops
#import utils
import pandas as pd
import multiprocessing as mp
total_proc = 2

#train, test = utils.load(0)

col_di = {'question1':'q1','question2':'q2'}

train = pd.read_csv('../input/train.csv.zip', usecols=['id','question1','question2']).rename(columns=col_di)
test  = pd.read_csv('../input/test.csv.zip', usecols=['test_id','question1','question2']).rename(columns=col_di)

train.dropna(inplace=True)
test.dropna(inplace=True)



#==============================================================================
# def
#==============================================================================

def get_tag(s):
    s = nlp(s)
    return [(token.text, token.tag_) for token in s if token.tag_ != 'SP']

def named_entities(s):
    s = nlp(s)
    return [(ent.text, ent.label_) for ent in s.ents]


def main(df):
    """
    
    
    """
    df['q1'] = df.q1.map(utils.preprocessing)
    df['q2'] = df.q2.map(utils.preprocessing)
    df['q1_wt'] = df.q1.map(get_tag)
    df['q2_wt'] = df.q2.map(get_tag)
    df['q1_named_entities'] = df.q1.map(named_entities)
    df['q2_named_entities'] = df.q2.map(named_entities)
    df['q1_noun_chunks'] = df.q1.map(utils.noun_chunks)
    df['q2_noun_chunks'] = df.q2.map(utils.noun_chunks)
    
    df.drop(['q1','q2'], axis=1, inplace=1)
    
    
    if 'id' in df.columns:
        df[['id', 'q1_wt', 'q2_wt']].to_pickle('../input/mk/train_tag.p')
        df[['id', 'q1_named_entities', 'q2_named_entities']].to_pickle('../input/mk/train_ent.p')
        df[['id', 'q1_noun_chunks', 'q2_noun_chunks']].to_pickle('../input/mk/train_chunks.p')
    else:
        df[['test_id', 'q1_wt', 'q2_wt']].to_pickle('../input/mk/test_tag.p')
        df[['test_id', 'q1_named_entities', 'q2_named_entities']].to_pickle('../input/mk/test_ent.p')
        df[['test_id', 'q1_noun_chunks', 'q2_noun_chunks']].to_pickle('../input/mk/test_chunks.p')
    return


"""

tmp = train.sample(1000)
tmp['q1_wt'] = tmp.q1.map(get_tag)
tmp['q1_named_entities'] = tmp.q1.map(named_entities)
tmp['q1_noun_chunks'] = tmp.q1.map(noun_chunks)


"""

#==============================================================================
# 
#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, [train, test])



print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))




