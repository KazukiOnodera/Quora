# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:16:56 2017

@author: konodera
"""

from collections import Counter
import numpy as np
import utils
stops = set(['a', 'an'])


#w2v_google = utils.load_vec('w2v-google')
#w2v_quora = utils.load_vec('w2v-quora-0')



train, test = utils.load(8, 1)

#==============================================================================
# def
#==============================================================================
def main(df):
#    df.drop(['q1_set_diff', 'q2_set_diff','_or_'], axis=1, inplace=1)
    df.drop(['_or_'], axis=1, inplace=1)
#    df.set_diff = df.set_diff.map(lambda x: sorted(set(x) - stops))
    df.set_diff = df.set_diff.map(lambda x: sorted(x))

def word(w):
    return train.ix[(train.q1.map(lambda x: w in x)) | (train.q2.map(lambda x: w in x))]

main(train); main(test)

raise

train_1vs1 = train[(train.q1_set_diff.map(len)==1) & (train.q2_set_diff.map(len)==1)]





train_diff0 = train[train.set_diff.map(len)==0]
test_diff0 = test[test.set_diff.map(len)==0]

train_diff1 = train[train.set_diff.map(len)==1]
test_diff1 = test[test.set_diff.map(len)==1]

train_diff2 = train[train.set_diff.map(len)==2]
test_diff2 = test[test.set_diff.map(len)==2]



tmp = word('.com')



df = train.sample(999)

word_tbl = Counter(sum(df.set_diff,[]))


np.random.choice(list(word_tbl.keys()))


raise


train_1 = train.ix[train.is_duplicate==1]




df = train.ix[train.set_diff.map(lambda x: len(x))<5]

words = sum(df.set_diff.tolist(),[])
words = list(set(words))

w = np.random.choice(words)
print(w)
train[train.set_diff.map(lambda x: w in x)].is_duplicate.mean()




w = '.com'
df1=train[train.set_diff.map(lambda x: w in ' '.join(x))]



print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

