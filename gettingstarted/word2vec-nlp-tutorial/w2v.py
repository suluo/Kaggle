#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : w2v.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-09
# Last Modified: 2018-03-28 15:13:08
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
import sys
import jieba
from gensim.models import word2vec, Word2Vec
import pandas as pd
from tqdm import tqdm
import time
import logging
import logging.config
logging.config.fileConfig('../conf/logging.conf')
logger = logging.getLogger(__file__)
import spacy
spacy_en = spacy.load('en')


def tokenizer(text):
    return [tok for tok in spacy_en(text)]


def load_tsv(tsv, filename):
    train = pd.read_csv(tsv, delimiter='\t')
    with open(filename, "a") as f:
        for review in tqdm(train['review']):
            seglist = jieba.cut(review)
            f.write(' '.join(seglist))
            f.write("\n")


def train():
    t0 = time.time()
    filename = './data/seg20180327.txt'
    if not os.path.exists(filename):
        for tsv in ['labeledTrainData.tsv', 'unlabeledTrainData.tsv', 'testData.tsv']:
            logger.info("loading %s ...." % tsv)
            load_tsv('./data/' + tsv, filename)
    sents = word2vec.Text8Corpus(filename)
    t1 = time.time()
    logger.info("load text taks %s" % (time.time()-t0))

    model_path = './data/model.w2v'
    if not os.path.exists(model_path):
        num_features, num_workers = 300, 4
        min_word_count, context = 20, 10
        downsampling = 1e-3
        model = word2vec.Word2Vec(
            sents, workers=num_workers,
            size=num_features, min_count=min_word_count,
            window=context, sample=downsampling
        )
        model.init_sims(replace=True)
    else:
        model = Word2Vec.load(model_path)
        # model.save_word2vec_foramt(output_vec, binary=False)
        model.build_vocab(sents, update=True)
        model.train(sents, total_examples=model.corpus_count, epochs=model.iter)
    # 生成的词典
    # model.vocab
    logger.info('w2v train taks %s' % (time.time()-t1))
    model.save('./data/model.w2v')


def main(num):
    train()
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)
