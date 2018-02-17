# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 23:36:19 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid))

import gensim

gloveFile_in = '/Users/konodera/Quora/nlp_source/glove.twitter.27B.200d.txt'
gloveFile_out = '/Users/konodera/Quora/nlp_source/glove.twitter.27B.200d'

gensim.scripts.glove2word2vec.glove2word2vec(gloveFile_in, gloveFile_out)

# gzip

# check
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/konodera/Quora/nlp_source/glove.twitter.27B.200d.gz')

model.similarity('security', 'protection')

print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

