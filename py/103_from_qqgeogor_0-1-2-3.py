# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:30:04 2017

@author: konodera

https://github.com/qqgeogor/kaggle_quora_benchmark

"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import utils
import distance
import multiprocessing as mp
total_proc = 4
#==============================================================================
# def
#==============================================================================

def str_jaccard(str1, str2):
    str1_list = str1.split(" ")
    str2_list = str2.split(" ")
    res = distance.jaccard(str1_list, str2_list)
    return res

# shortest alignment
def str_levenshtein_1(str1, str2):
    res = distance.nlevenshtein(str1, str2,method=1)
    return res

# longest alignment
def str_levenshtein_2(str1, str2):
    res = distance.nlevenshtein(str1, str2,method=2)
    return res

def str_sorensen(str1, str2):
    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.sorensen(str1_list, str2_list)
    return res

def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / len(a)


def main(p):
    
    train, test = utils.load(p)
    
    train['qqgeogor_jaccard-{}'.format(p)] = train.apply(lambda x: str_jaccard(x['q1'], x['q2']), axis=1)
    train['qqgeogor_levenshtein_1-{}'.format(p)] = train.apply(lambda x: str_levenshtein_1(x['q1'], x['q2']), axis=1)
    train['qqgeogor_levenshtein_2-{}'.format(p)] = train.apply(lambda x: str_levenshtein_2(x['q1'], x['q2']), axis=1)
    train['qqgeogor_sorensen-{}'.format(p)] = train.apply(lambda x: str_sorensen(x['q1'], x['q2']), axis=1)
    train['qqgeogor_set_intersection-{}'.format(p)] = train.apply(lambda x: calc_set_intersection(x['q1'], x['q2']), axis=1)
    
    test['qqgeogor_jaccard-{}'.format(p)] = test.apply(lambda x: str_jaccard(x['q1'], x['q2']), axis=1)
    test['qqgeogor_levenshtein_1-{}'.format(p)] = test.apply(lambda x: str_levenshtein_1(x['q1'], x['q2']), axis=1)
    test['qqgeogor_levenshtein_2-{}'.format(p)] = test.apply(lambda x: str_levenshtein_2(x['q1'], x['q2']), axis=1)
    test['qqgeogor_sorensen-{}'.format(p)] = test.apply(lambda x: str_sorensen(x['q1'], x['q2']), axis=1)
    test['qqgeogor_set_intersection-{}'.format(p)] = test.apply(lambda x: calc_set_intersection(x['q1'], x['q2']), axis=1)
    
    utils.to_csv(train, test, 'f103-{}'.format(p))
    
    return

#==============================================================================


pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))

#main(9)


print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

