#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : w2v.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-09
# Last Modified: 2018-03-09 14:34:26
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
import sys
from gensim.models import word2vec, Word2Vec
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)
# 准备使用nltk的tokenizer对影评中的英文句子进行分割

def train(corpora):
    num_features = 300
    num_workers = 4
    min_word_count=20
    context = 10
    downsampling = le-3
    model = word2vec.Word2Vec(
        corpora, workers=num_workers,
        size=num_features, min_count=min_word_count,
        window=context, sample=downsampling
    )
    model.init_sims(replace=True)
    model.save('../data/300features_20min_10context.w2v')


def load_w2v():
    model = Word2Vec.load("../data/300features_20min_10context.w2v")


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


