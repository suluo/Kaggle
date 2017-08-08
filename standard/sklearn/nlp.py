#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name: nlp.py
# Purpose:
#            xgboost install:https://zhuanlan.zhihu.com/p/23996104
# Creation Date: 2017-07-31
# Last Modified: 2017-08-01 13:50:50
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division
import logging
import logging.handlers
#import traceback
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import nltk
tokens = nltk.word_tokenize(sentence)
pos_tag = nltk.tag.pos_tag(tokens)

from gensim.models import word2vec
num_features = 300 # 词向量维度
min_word_count = 20 # 词汇最小频度
num_workers = 2 # cpu核数
context = 5
downsampling = 1e-3

model = word2vec.Word2vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling)
# 当前训练最终版，加快训练速度
model.init_sims(replace=True)

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


