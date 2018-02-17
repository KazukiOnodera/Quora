# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:48:18 2017

@author: Kazuki

nohup python 001_perfect.py &

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
import multiprocessing as mp
total_proc = 2



#==============================================================================
# def
#==============================================================================

def rem_a(s1, s2):
    s1 = s1.replace(' a ', ' ').replace(' an ', ' ')
    s2 = s2.replace(' a ', ' ').replace(' an ', ' ')
    return (s1 == s2)*1
    
def rem_the(s1, s2):
    s1 = s1.replace(' the ', ' ')
    s2 = s2.replace(' the ', ' ')
    return (s1 == s2)*1

    
def rem_a_the(s1, s2):
    s1 = s1.replace(' the ', ' ').replace(' a ', ' ').replace(' an ', ' ')
    s2 = s2.replace(' the ', ' ').replace(' a ', ' ').replace(' an ', ' ')
    return (s1 == s2)*1
    
def rem_my(s1, s2):
    s1 = s1.replace(' my ', ' ').replace(' your ', ' ')
    s2 = s2.replace(' my ', ' ').replace(' your ', ' ')
    return (s1 == s2)*1
    
def lower(s1, s2):
    return (s1.lower() == s2.lower())*1

def rem_lower(s1, s2):
    s1 = s1.replace(' the ', ' ').replace(' a ', ' ').replace(' an ', ' ').replace(' my ', ' ').replace(' your ', ' ')
    s2 = s2.replace(' the ', ' ').replace(' a ', ' ').replace(' an ', ' ').replace(' my ', ' ').replace(' your ', ' ')
    return (s1.lower() == s2.lower())*1

def part_fit(s1, s2):
    s1 = s1.replace('!','').replace('?','').replace('.','').strip()
    s2 = s2.replace('!','').replace('?','').replace('.','').strip()
    if s1 in s2 or s2 in s2:
        return 1
    return 0

def main(args):
    df = args[0]
    suf = args[1]
    
    df['perfect'+suf] = (df['q1'] == df['q2'])*1
    
    if suf not in ['-stop','-stst']:
        df['rem_a'+suf] = df.apply(lambda x: rem_a(x['q1'], x['q2']), axis=1)
        df['rem_the'+suf] = df.apply(lambda x: rem_the(x['q1'], x['q2']), axis=1)
        df['rem_a_the'+suf] = df.apply(lambda x: rem_a_the(x['q1'], x['q2']), axis=1)
        df['rem_my'+suf] = df.apply(lambda x: rem_my(x['q1'], x['q2']), axis=1)
        df['rem_lower'+suf] = df.apply(lambda x: rem_lower(x['q1'], x['q2']), axis=1)
        
    df['lower'+suf] = df.apply(lambda x: lower(x['q1'], x['q2']), axis=1)
    
    df['part_fit'+suf] = df.apply(lambda x: part_fit(x['q1'], x['q2']), axis=1)
    
    
    
    return df


#==============================================================================

train, test = utils.load(0)
pool = mp.Pool(total_proc)
suf = ''
callback = pool.map(main, [(train,suf), (test,suf)])
utils.to_csv(callback[0], callback[1], 'f001-0')

train, test = utils.load(1)
suf = '-stem'
callback = pool.map(main, [(train,suf), (test,suf)])
utils.to_csv(callback[0], callback[1], 'f001-1')

train, test = utils.load(2)
suf = '-stop'
callback = pool.map(main, [(train,suf), (test,suf)])
utils.to_csv(callback[0], callback[1], 'f001-2')

train, test = utils.load(3)
suf = '-stst'
callback = pool.map(main, [(train,suf), (test,suf)])
utils.to_csv(callback[0], callback[1], 'f001-3')

#train, test = utils.load(10)
#train = main(train, '-syn0')
#test = main(test, '-syn0')
#utils.to_csv(train, test, 'f001-10')



print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

