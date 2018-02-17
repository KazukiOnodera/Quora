# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:08:22 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
stops = utils.stops

#==============================================================================
# def
#==============================================================================
def get_VO(s):
    """
    """
    
    verb = obje = None
    
    if s.startswith('How') and len(s.split())>5:
        wt = utils.get_tag(s)
        # How can I ~
        # How do you ~
        if wt[1][1] in ['VBP','WRB', 'VB','MD'] and wt[2][1] == 'PRP' and 'VB' in wt[3][1]:
            verb = wt[3][0]
            obje = utils.find_noun_after_prep(verb, s, False)
            
            return verb, obje
        return verb, obje
    
    elif s.startswith('Is it possible to to') and len(s.split())>5:
        s_ = s.split()
        ix = s_.index('to') + 1
        verb = s_[ix]
        obje = utils.find_noun_after_prep(verb, s, False)
        
        return verb, obje
    
    elif (' ways to ' in s or ' way to ' in s ) and len(s.split())>5:
        s_ = s.split()
        ix = s_.index('to') + 1
        verb = s_[ix]
        obje = utils.find_noun_after_prep(verb, s, False)
        
        return verb, obje
    
    if len(s.split())>3:
        wt = utils.get_tag(s)
        if wt[0][1] in ['MD'] and wt[1][1] == 'PRP' and 'VB' in wt[2][1]:
            verb = wt[2][0]
            obje = utils.find_noun(wt)
            
            return verb, obje
    
    
    return verb, obje

def extract_V(wt):
    return [w for w,t in wt if w.lower() not in stops and 'VB' in t]

def extract_0(wt):
    return [w for w,t in wt if w.lower() not in stops and 'NN' in t]

def extract_V(wt):
    return [w for w,t in wt if w.lower() not in stops and 'VB' in t]





#==============================================================================

train, test = utils.load(4,1)


train['q1_v'] = train.q1.map(extract_V)
train['q2_v'] = train.q2.map(extract_V)

train['q1_o'] = train.q1.map(extract_0)
train['q2_o'] = train.q2.map(extract_0)

train_ori, test_ori = utils.load(0)

import pandas as pd
train_ = pd.merge(train_ori, train[['is_duplicate','id','q1_v','q2_v','q1_o','q2_o']], 
                 on='id', how='left')



#
#train['q1_vo'] = train.q1.map(get_VO)
#
#train['q1_wt'] = train.q1.map(utils.get_tag)















