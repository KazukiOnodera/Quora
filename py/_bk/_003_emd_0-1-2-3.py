# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 00:34:30 2017

@author: Kazuki
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import gc
import numpy as np
from nltk import word_tokenize
import utils
stops = utils.stops
import gensim
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

#==============================================================================
# def
#==============================================================================
def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
#    stops = stopwords.words('english')
    s1 = [w for w in s1 if w not in stops]
    s2 = [w for w in s2 if w not in stops]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
#    stops = stopwords.words('english')
    s1 = [w for w in s1 if w not in stops]
    s2 = [w for w in s2 if w not in stops]
    return norm_model.wmdistance(s1, s2)

def sent2vec(s):
    words = str(s).lower()#.decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stops]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def main(df, suf):
    
    print('start calc wmd!')
    
    df['wmd'+suf] = df.apply(lambda x: wmd(x['q1'], x['q2']), axis=1)
    
    df['norm_wmd'+suf] = df.apply(lambda x: norm_wmd(x['q1'], x['q2']), axis=1)
    
    print('start calc distance!')
    
    question1_vectors = np.zeros((df.shape[0], 300))
    for i, q in tqdm(enumerate(df['q1'].values)):
        question1_vectors[i, :] = sent2vec(q)
    
    question2_vectors  = np.zeros((df.shape[0], 300))
    for i, q in tqdm(enumerate(df['q2'].values)):
        question2_vectors[i, :] = sent2vec(q)
    
    df['cosine_dist'+suf] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    df['cityblock_dist'+suf] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    df['jaccard_dist'+suf] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    df['canberra_dist'+suf] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    df['euclidean_dist'+suf] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    df['minkowski_dist'+suf] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    
    df['braycurtis_dist'+suf] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
        
    print('start calc vec!')
    
    df['skew_q1vec'+suf] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    df['skew_q2vec'+suf] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    df['kur_q1vec'+suf] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    df['kur_q2vec'+suf] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
    
    return df

#==============================================================================

model = gensim.models.KeyedVectors.load_word2vec_format('../nlp_source/GoogleNews-vectors-negative300.bin.gz', binary=True)

norm_model = gensim.models.KeyedVectors.load_word2vec_format('../nlp_source/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)




train, test = utils.load(0)
train = main(train, '')
test = main(test, '')
utils.to_csv(train, test, 'f003-0')
del train, test; gc.collect()

train, test = utils.load(1)
train = main(train, '-stem')
test = main(test, '-stem')
utils.to_csv(train, test, 'f003-1')
del train, test; gc.collect()

train, test = utils.load(2)
train = main(train, '-stop')
test = main(test, '-stop')
utils.to_csv(train, test, 'f003-2')
del train, test; gc.collect()

train, test = utils.load(3)
train = main(train, '-stst')
test = main(test, '-stst')
utils.to_csv(train, test, 'f003-3')
del train, test; gc.collect()





print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))






