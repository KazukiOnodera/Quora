# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 08:56:28 2017

@author: Kazuki


from https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py

nohup python 100_from_abish.py &

"""

import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

from fuzzywuzzy import fuzz
import utils
import gc

#==============================================================================
# def
#==============================================================================


def main(df, suf):
    
    df['len_q1'+suf] = df['q1'].apply(lambda x: len(str(x)))
    df['len_q2'+suf] = df['q2'].apply(lambda x: len(str(x)))
    df['diff_len'+suf] = df['len_q1'+suf] - df['len_q2'+suf]
    df['len_char_q1'+suf] = df['q1'].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    df['len_char_q2'+suf] = df['q2'].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    df['len_word_q1'+suf] = df['q1'].apply(lambda x: len(str(x).split()))
    df['len_word_q2'+suf] = df['q2'].apply(lambda x: len(str(x).split()))
    df['common_words'+suf] = df.apply(lambda x: len(set(str(x['q1']).lower().split()).intersection(set(str(x['q2']).lower().split()))), axis=1)
    df['fuzz_qratio'+suf] = df.apply(lambda x: fuzz.QRatio(str(x['q1']), str(x['q2'])), axis=1)
    df['fuzz_WRatio'+suf] = df.apply(lambda x: fuzz.WRatio(str(x['q1']), str(x['q2'])), axis=1)
    df['fuzz_partial_ratio'+suf] = df.apply(lambda x: fuzz.partial_ratio(str(x['q1']), str(x['q2'])), axis=1)
    df['fuzz_partial_token_set_ratio'+suf] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['q1']), str(x['q2'])), axis=1)
    df['fuzz_partial_token_sort_ratio'+suf] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['q1']), str(x['q2'])), axis=1)
    df['fuzz_token_set_ratio'+suf] = df.apply(lambda x: fuzz.token_set_ratio(str(x['q1']), str(x['q2'])), axis=1)
    df['fuzz_token_sort_ratio'+suf] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['q1']), str(x['q2'])), axis=1)
    
    return df
#==============================================================================


train, test = utils.load(0)
train = main(train, '')
test = main(test, '')
utils.to_csv(train, test, 'f100-0')
del train, test; gc.collect()

train, test = utils.load(1)
train = main(train, '-stem')
test = main(test, '-stem')
utils.to_csv(train, test, 'f100-1')
del train, test; gc.collect()

train, test = utils.load(2)
train = main(train, '-stop')
test = main(test, '-stop')
utils.to_csv(train, test, 'f100-2')
del train, test; gc.collect()

train, test = utils.load(3)
train = main(train, '-stst')
test = main(test, '-stst')
utils.to_csv(train, test, 'f100-3')
del train, test; gc.collect()

#train, test = utils.load(9)
#train = main(train, '-syn1')
#test = main(test, '-syn1')
#utils.to_csv(train, test, 'f100-3')
#del train, test; gc.collect()




print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))










