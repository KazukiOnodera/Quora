# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:28:12 2017

@author: Kazuki

nohup python 004_how.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

from nltk import stem
pt = stem.PorterStemmer().stem
import utils
stops = utils.stops
stops_noun = set(['best','way','ways','one'])
import pandas as pd
import multiprocessing as mp
total_proc = 2
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)

train, test = utils.load(0, 1)

#==============================================================================
# def
#==============================================================================
def len_match(s1, s2, ret):
    
    s1_ = s1.split()
    s2_ = s2.split()
    s1_wt = utils.get_tag(s1)
    s2_wt = utils.get_tag(s2)
    
    # How [do | can | will] [you | I | we] [verb] ~ ?
    if s1_wt[1][1] in ['VBP','VB','MD'] and s2_wt[1][1] in ['VBP','VB','MD'] and\
    s1_wt[3][1] == s2_wt[3][1] == 'VB' and\
    s1_wt[2][1] == s2_wt[2][1] == 'PRP':
        ret += '-vsv'
            
        # head 3 words match!!!
        if s1_[:3] == s2_[:3]:
            ret += '-h3'
            return ret
            
        # head 2 words match!!!
        elif s1_[:2] == s2_[:2]:
            ret += '-h2'
            return ret
            
        elif s1_[2:] == s2_[2:]:
            """
            s1_ = 'How can you improve intelligence ?'.split()
            s2_ = 'How do you improve intelligence ?'.split()
            """
            return ret+'1'
            
        elif s1_[1] == s2_[1] and s1_[3:] == s2_[3:]:
            """
            s1_ = 'How do I improve intelligence ?'.split()
            s2_ = 'How do you improve intelligence ?'.split()
            """
            return ret+'2'
        
        elif s1_[3:] == s2_[3:]:
            """
            s1_ = 'How can I improve intelligence ?'.split()
            s2_ = 'How do you improve intelligence ?'.split()
            """
            return ret+'12'
        else:
            return ret+'N'
    else:
        ret += '-oth'
        return ret

def find_nouns(s1_wt, s2_wt, ret):
    try:
        # search nearest noun(Objective)
        n1 = [w for w,t in s1_wt[4:] if 'NN' in t][0]
        n2 = [w for w,t in s2_wt[4:] if 'NN' in t][0]
        if n1 == n2:
            return ret+'-a'
        
        elif pt(n1) == pt(n2):
            return ret+'-b'
            
        elif pt(n1.lower()) == pt(n2.lower()):
            return ret+'-c'
            
        else:
            return ret+'-N'
    except:
        return ret    


def how(s1, s2):
    """
    s1 ='How do you imagine future?'
    s2 ='How can I imagine future?'
    
    utils.get_tag('How can you improve your intelligence ? ')
    utils.get_tag('How will I contact a genuine hacker ? ')
    utils.get_tag('How do you connect to wifi using TAILs ? ')
    
    """
    ret = ''
    # double how
    if s1.startswith('How') and s2.startswith('How'):
        ret += 'dh'
        
        rem = [' a ',' an ', '?', ' the ', ' my ', ' your ']
        s1_rem = s1; s2_rem = s2
        s1_ = s1.split(); s2_ = s2.split()
        for r in rem:
            s1_rem = s1_rem.replace(r,' ')
            s2_rem = s2_rem.replace(r,' ')
            
        if s1 == s2:
            return ret+'p1'
            
        elif s1_rem == s2_rem:
            return ret+'p2'
        
        elif len(s1.split())<4 or len(s2.split())<4:
            return ret+'s'
        
        
        s1_wt = utils.get_tag(s1)
        s2_wt = utils.get_tag(s2)
        
        # How [do | can | will] [you | I | we] [verb] ~ ?
        if s1_wt[1][1] in ['VBP','VB','MD'] and s2_wt[1][1] in ['VBP','VB','MD'] and\
        s1_wt[3][1] == s2_wt[3][1] == 'VB' and\
        s1_wt[2][1] == s2_wt[2][1] == 'PRP' and len(s1_)>4 and len(s2_)>4:
            ret += '-vsv'
            
            # head 4 words match!!!
            if s1_[:4] == s2_[:4]:
                ret += '-h4'
                s1_wt = utils.get_tag(s1)
                s2_wt = utils.get_tag(s2)
                
                if len(s1_) == len(s2_) == 6:
                    ret += '-l6'
                    n1 = s1_[4]; n2 = s2_[4]
                    if pt(n1) == pt(n2):
                        return ret+'-a'
                    elif pt(n1.lower()) == pt(n2.lower()):
                        return ret+'-b'
                    else:
                        return ret+'-N'
                
                return find_nouns(s1_wt, s2_wt, ret)
        
            # head 3 words match!!!
            elif s1_[:3] == s2_[:3]:
                ret += '-h3'
                return ret
            
            # head 2 and 4- match
            elif s1_[1] == s2_[1] and s1_[3:] == s2_[3:]:
                """
                s1_ = 'How do I improve intelligence ?'.split()
                s2_ = 'How do you improve intelligence ?'.split()
                """
                ret += '-h2&4'
                
                return find_nouns(s1_wt, s2_wt, ret)
            
            # head 2 and 4- match
            elif s1_rem[1] == s2_rem[1] and s1_rem[3:] == s2_rem[3:]:
                """
                s1_ = 'How do I improve intelligence ?'.split()
                s2_ = 'How do you improve intelligence ?'.split()
                """
                ret += '-h2&4rem'
                return find_nouns(s1_wt, s2_wt, ret)
            
            # head 2 words match!!!
            elif s1_[:2] == s2_[:2]:
                ret += '-h2'
                return ret
                
            elif s1_[2:] == s2_[2:]:
                """
                s1_ = 'How can you improve intelligence ?'.split()
                s2_ = 'How do you improve intelligence ?'.split()
                """
                ret += '-2'
                return find_nouns(s1_wt, s2_wt, ret)
                
            
            elif s1_[3:] == s2_[3:]:
                """
                s1_ = 'How can I improve intelligence ?'.split()
                s2_ = 'How do you improve intelligence ?'.split()
                """
                ret += '-23'
                return find_nouns(s1_wt, s2_wt, ret)
            
            elif s1_[2:4] == s2_[2:4]:
                """
                s1_ = 'How should I prepare for the GATE CE ?'.split()
                s2_ = 'How do I prepare for the GATE CE 2018 ?'.split()
                """
                ret += '-1'
                
                return find_nouns(s1_wt, s2_wt, ret)
                
#            elif ret :
#                return ret
#                
#            elif ret :
#                return ret
#                
#            elif ret :
#                return ret
#                
#            elif ret :
#                return ret
#                
#            elif ret :
#                return ret
                
            else:
                return ret+'-N'
            
        else:
            ret += '-nvsv'
            return ret
            
        return ret # dh
        
    else:
        return ret
    return ret


def extract(s, n):
    try:
        return s.split('-')[n]
    except:
        return ''

def how_V(s):
    wt = utils.get_tag(s)
    vlist = [w for w,t in wt if w.lower() not in stops and t.startswith('V')]
    if len(vlist)==0:
        return ['is']
    return vlist
#    if s.startswith('How') or s.startswith('how'):
#        wt = utils.get_tag(s)
#        vlist = [w for w,t in wt if w not in stops and t.startswith('V')]
#        return vlist
#    else:
#        return []

def noun(s):
    s = sum([w.split() for w in utils.noun_chunks(s)], [])
    s = [w for w in s if w.lower() not in stops_noun]
    return s

def main(p):
    """
    w='How'
    df = train[train.q1.str.startswith(w)|train.q2.str.startswith(w)].sample(999)
    df['q1_how_V'] = df.q1.map(how_V)
    df['q2_how_V'] = df.q2.map(how_V)
    df['q1_noun'] = df.q1.map(noun)
    df['q2_noun'] = df.q2.map(noun)
    
    """
    
    if p==0:
        df = train
    elif p==1:
        df = test
    
    df['how'] = df.apply(lambda x: how(x['q1'], x['q2']), axis=1)
    df['how0'] = df.how.map(lambda x: extract(x,0))
    df['how1'] = df.how.map(lambda x: extract(x,1))
    df['how2'] = df.how.map(lambda x: extract(x,2))
    
    df = pd.get_dummies(df, columns=['how', 'how0', 'how1', 'how2'])
    
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





