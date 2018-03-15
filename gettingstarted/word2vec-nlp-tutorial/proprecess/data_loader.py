#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : data_loader.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-13 19:23:38
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
import codecs
import numpy as np
import pandas as pd
import random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from torchtext import data
from bs4 import BeautifulSoup
import torch.utils.data as Data
import re
SEED = 1


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

import spacy
spacy_en = spacy.load('en')


# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
class dataset():
    def __init__(self,):
        super(dataset, self).__init__()
        self.vocab_size = 0
        self.label_size = 0
        self.batch_size = 1

    def tokenizer(text): # create a tokenizer function
        # 返回 a list of <class 'spacy.tokens.token.Token'>
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def load_train(self, batch_size):
        x_train, y_train = self.load_train_vector()
        print (len(x_train), len(y_train))

        x_train = np.array(x_train, dtype=int)
        y_train = np.array(y_train, dtype=int)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        train_index = 20000
        train_torch_dataset = Data.TensorDataset(
            data_tensor=x_train[:train_index],
            target_tensor=y_train[:train_index]
        )
        train_loader = Data.DataLoader(
            dataset=train_torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        dev_torch_dataset = Data.TensorDataset(
            data_tensor=x_train[train_index+1:],
            target_tensor=y_train[train_index+1:]
        )
        dev_loader = Data.DataLoader(
            dataset=dev_torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        return train_loader, dev_loader

    def prepare_sequence(self, seq, to_ix):
        var = torch.LongTensor([to_ix[w] for w in seq.split(' ')])
        return var

    def prepare_label(self, label, label_to_ix):
        var = torch.LongTensor([label_to_ix[label]])
        return var

    def load_train_vector(self):
        print ("loading vocab ...")
        word_vocab, label_vocab = self.load_vocab()
        self.vocab_size, self.label_size = len(word_vocab), len(label_vocab)

        train = pd.read_csv('./data/labeledTrainData.tsv', delimiter="\t")
        x_train = []
        print ("loading train data start...")
        for review in tqdm(train['review']):
            # x_train.append(" ".join(self.review_to_text(review, False)))
            x_train.append([word_vocab[w] for w in self.review_to_text(review, False)])
        y_train = [label_vocab[w] for w in train['sentiment']]
        return x_train, y_train

    def load_vocab(self):
        train = pd.read_csv('./data/labeledTrainData.tsv', delimiter="\t")
        corpora = []
        print ('loading vocab train ...')
        for review in tqdm(train['review']):
            corpora.append(' '.join(self.review_to_text(review, False)))
            # corpora.extend([" ".join(sent) for sent in self.review_to_sentences(review)])

        unlabel_train = pd.read_csv("./data/unlabeledTrainData.tsv", delimiter="\t", quoting=3)
        print ('loading vocab unlabel ...')
        for review in tqdm(unlabel_train['review']):
            corpora.append(' '.join(self.review_to_text(review, False)))

        word_to_ix = self.build_token_to_ix(corpora)
        label_to_ix = {0: 0, 1: 1}
        print ('vocab size:',len(word_to_ix),
               'label size:',len(label_to_ix))
        return word_to_ix, label_to_ix

    def review_to_text(self, review, remove_stopwords=True):
        raw_text = BeautifulSoup(review, 'lxml').get_text()
        letters = re.sub('[^a-zA-Z]', ' ', raw_text)
        words = letters.lower().split()
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        return words

    def review_to_sentences(self, review):
        raw_sentences = sent_tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.review_to_text(raw_sentence, False))
        return sentences

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


def load_train_batch(text_field, label_field, batch_size):
    print ("loading train data start ...")
    train_data, dev_data = dataset.splits(text_field, label_field)
    for i in train_data:
        print (i)
    print (type(train_data), len(dev_data))
    text_field.build_vocab(train_data + dev_data, vectors="glove.6B.100d")
    print ("building batches ...")
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        sort_key=lambda x: len(x.text),
        repeat=False, device=-1
    )
    return train_iter, dev_iter


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


