# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 07:29:56 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))

import pandas as pd
#import numpy as np
import os
import utils
import multiprocessing as mp
total_proc = 2
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import LabeledSentence
#==============================================================================
# param
#==============================================================================
WORD2VEC_MODEL_DIR = '../nlp_source/w2v/Quora/'
DOC2VEC_MODEL_DIR = '../nlp_source/d2v/Quora/'

EMBEDDING_ALPHA = 0.025
EMBEDDING_LEARNING_RATE_DECAY = 0.5
EMBEDDING_N_EPOCH = 5
EMBEDDING_MIN_COUNT = 3
EMBEDDING_DIM = 100
EMBEDDING_WINDOW = 5
EMBEDDING_WORKERS = 10

token_pattern = " "

w2v_param = {
            "alpha": EMBEDDING_ALPHA,
            "learning_rate_decay": EMBEDDING_LEARNING_RATE_DECAY,
            "n_epoch": EMBEDDING_N_EPOCH,
            "sg": 1,
            "hs": 1,
            "min_count": EMBEDDING_MIN_COUNT,
            "size": EMBEDDING_DIM,
            "sample": 0.001,
            "window": EMBEDDING_WINDOW,
            "workers": EMBEDDING_WORKERS,
            }
            
d2v_param = {
            "alpha": EMBEDDING_ALPHA,
            "learning_rate_decay": EMBEDDING_LEARNING_RATE_DECAY,
            "n_epoch": EMBEDDING_N_EPOCH,
            "sg": 1, # not use
            "dm": 1,
            "hs": 1,
            "min_count": EMBEDDING_MIN_COUNT,
            "size": EMBEDDING_DIM,
            "sample": 0.001,
            "window": EMBEDDING_WINDOW,
            "workers": EMBEDDING_WORKERS,
            }
#==============================================================================
# nlp_utils
#==============================================================================

import re


def _tokenize(text, token_pattern=" "):
    # token_pattern = r"(?u)\b\w\w+\b"
    # token_pattern = r"\w{1,}"
    # token_pattern = r"\w+"
    # token_pattern = r"[\w']+"
    if token_pattern == " ":
        # just split the text into tokens
        return text.split(" ")
    else:
        token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
        group = token_pattern.findall(text)
        return group

#==============================================================================
# w2v
#==============================================================================

class DataFrameSentences(object):
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns

    def __iter__(self):
        for column in self.columns:
            for sentence in self.df[column]:
                tokens = _tokenize(sentence, token_pattern)
                yield tokens


class DataFrameWord2Vec:
    def __init__(self, df, columns, model_param):
        self.df = df
        self.columns = columns
        self.model_param = model_param
        self.model = Word2Vec(sg=self.model_param["sg"], 
                                hs=self.model_param["hs"], 
                                alpha=self.model_param["alpha"],
                                min_alpha=self.model_param["alpha"],
                                min_count=self.model_param["min_count"], 
                                size=self.model_param["size"], 
                                sample=self.model_param["sample"], 
                                window=self.model_param["window"], 
                                workers=self.model_param["workers"])

    def train(self):
        # build vocabulary
        self.sentences = DataFrameSentences(self.df, self.columns)
        self.model.build_vocab(self.sentences)
        # train for n_epoch
        for i in range(self.model_param["n_epoch"]):
            self.sentences = DataFrameSentences(self.df, self.columns)
            self.model.train(self.sentences)
            self.model.alpha *= self.model_param["learning_rate_decay"]
            self.model.min_alpha = self.model.alpha
        return self

    def save(self, model_dir, model_name):
        fname = os.path.join(model_dir, model_name)
        self.model.save(fname)



def train_word2vec_model(df, columns):
    model_dir = WORD2VEC_MODEL_DIR
    model_name = "Quora-word2vec-2-D%d-min_count%d.model"%(
                    w2v_param["size"], w2v_param["min_count"])

    word2vec = DataFrameWord2Vec(df, columns, w2v_param)
    word2vec.train()
    word2vec.save(model_dir, model_name)

#==============================================================================
# d2v
#==============================================================================
class DataFrameLabelSentences(object):
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
        self.cnt = -1
        self.sent_label = {}

    def __iter__(self):
        for column in self.columns:
            for sentence in self.df[column]:
                if not sentence in self.sent_label:
                    self.cnt += 1
                    self.sent_label[sentence] = "SENT_%d"%self.cnt
                tokens = _tokenize(sentence, token_pattern)
                yield LabeledSentence(words=tokens, tags=[self.sent_label[sentence]])


class DataFrameDoc2Vec(DataFrameWord2Vec):
    def __init__(self, df, columns, model_param):
        super().__init__(df, columns, model_param)
        self.model = Doc2Vec(dm=self.model_param["dm"], 
                                hs=self.model_param["hs"], 
                                alpha=self.model_param["alpha"],
                                min_alpha=self.model_param["alpha"],
                                min_count=self.model_param["min_count"], 
                                size=self.model_param["size"], 
                                sample=self.model_param["sample"], 
                                window=self.model_param["window"], 
                                workers=self.model_param["workers"])
    def train(self):
        # build vocabulary
        self.sentences = DataFrameLabelSentences(self.df, self.columns)
        self.model.build_vocab(self.sentences)
        # train for n_epoch
        for i in range(self.model_param["n_epoch"]):
            self.sentences = DataFrameLabelSentences(self.df, self.columns)
            self.model.train(self.sentences)
            self.model.alpha *= self.model_param["learning_rate_decay"]
            self.model.min_alpha = self.model.alpha
        return self

    def save(self, model_dir, model_name):
        fname = os.path.join(model_dir, model_name)
        self.model.save(fname)


def train_doc2vec_model(df, columns):

    model_dir = DOC2VEC_MODEL_DIR
    model_name = "Quora-doc2vec-2-D%d-min_count%d.model"%(
                    d2v_param["size"], d2v_param["min_count"])

    doc2vec = DataFrameDoc2Vec(df, columns, d2v_param)
    doc2vec.train()
    doc2vec.save(model_dir, model_name)

def main(p):
    if p==0:
        print('building word2vec')
        train_word2vec_model(df, columns)
        print('finish word2vec')
    elif p==1:
        print('building doc2vec')
        train_doc2vec_model(df, columns)
        print('finish doc2vec')
#==============================================================================
# main
#==============================================================================

train, test = utils.load(2)
df = pd.concat([train,test], ignore_index=1)
columns = ['q1','q2']


pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))






print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

