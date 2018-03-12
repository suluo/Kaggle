#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : data_loader.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-09 14:28:01
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
import sys
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)
import torch
import torch.autograd as autograd
import codecs
import pandas as pd
import random
from tqdm import tqdm
import torch.utils.data as Data
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

SEED = 1

# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
class dataset():
    def __init__(self):
        super(dataset, self).__init__()

    def load_train(self):
        train = pd.read_csv('./data/labeledTrainData.tsv', delimiter="\t")
        x_train = []
        print ("loading train data start...")
        for review in tqdm(train['review']):
            x_train.append(" ".join(self.review_to_text(review, False)))
        y_train = train['sentiment']

        train_data = [(x, y) for x, y in zip(x_train[:20000], train['sentiment'][:20000])]
        dev_data = [(x, y) for x, y in zip(x_train[20001:], train['sentiment'][20001:])]

        random.shuffle(train_data)
        random.shuffle(dev_data)

        word_to_ix = self.build_token_to_ix([s for s,_ in train_data + dev_data])
        label_to_ix = {0: 0, 1: 1}
        print('train:', len(train_data),
              'dev:',len(dev_data),
              'vocab size:',len(word_to_ix),
              'label size:',len(label_to_ix),
              'loading train data done!')
        return train_data, dev_data, word_to_ix, label_to_ix

    def load_w2v_data(self):
        unlabel_train = pd.read_csv("./data/unlabeledTrainData.tsv", delimiter="\t", quoting=3)
        corpora = []
        for review in tqdm(unlabel_train['review']):
            corpora += self.review_to_sentences(review.decode('utf8'), tokenizer)
        return corpora

    def review_to_text(self, review, remove_stopwords=True):
        raw_text = BeautifulSoup(review, 'lxml').get_text()
        letters = re.sub('[^a-zA-Z]', ' ', raw_text)
        words = letters.lower().split()
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        return words

    def review_to_sentences(self, review):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.review_to_text(raw_sentence, False))
        return sentences

    def prepare_sequence(self, seq, to_ix, cuda=False):
        var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
        return var

    def prepare_label(self, label,label_to_ix, cuda=False):
        var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
        return var

    def build_token_to_ix(self, sentences):
        token_to_ix = dict()
        # print(len(sentences))
        for sent in sentences:
            for token in sent.split(' '):
                if token not in token_to_ix:
                    token_to_ix[token] = len(token_to_ix)
        token_to_ix['<pad>'] = len(token_to_ix)
        return token_to_ix

    def build_label_to_ix(self, labels):
        label_to_ix = dict()
        for label in labels:
            if label not in label_to_ix:
                label_to_ix[label] = len(label_to_ix)

    def load_MR_data_batch():

        pass



def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


