# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:17:11 2017

@author: Kazuki
"""

import utils
import xgbextension as ex

cv, imp = utils.load_cv()


#ex.pltImp(imp, n=30)


w=' reading'
print('size: {:.3f}'.format(cv[cv.q1_ori.str.contains(w)|cv.q2_ori.str.contains(w)].shape[0]/cv.shape[0]))
df = cv[cv.q1_ori.str.contains(w)|cv.q2_ori.str.contains(w)].sample(999)
df['w_cnt'] = df.q1_ori.str.contains(w)*1 + df.q2_ori.str.contains(w)*1

raise





from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('../nlp_source/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)


s = 'What is the differences between Computer Science Engineering and Computer Engineering ?'
for w in s.split():
    print(w, (w in w2v)*1)


#==============================================================================
# set diff
#==============================================================================




"""

train = train.merge(cv[['id','yhat','d','q1_ori', 'q2_ori']], on='id')

train = train.merge(cv[['id','yhat','d']], on='id')

"""