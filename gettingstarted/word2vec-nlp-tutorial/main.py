#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : main.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-04-03 19:03:30
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import torch
from torchtext import data
from tqdm import tqdm
import pandas as pd
import random
import os
import time
import logging
import logging.config
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
from model.lstm import LSTM
# from proprecess import data_loader
import proprecess.data_loader_batch as data_loader
import train_batch as train

logging.config.fileConfig('./conf/logging.conf',
                          disable_existing_loggers=False)
logger = logging.getLogger(__file__)
import spacy
spacy_en = spacy.load("en")


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


# load word embedding
def load_my_vecs(path, vocab, freqs):
    word_vecs = {}
    with open(path, encoding="utf-8") as f:
        count = 0
        lines = f.readlines()[1:]
        for line in lines:
            values = line.split(" ")
            word = values[0]
            # word = word.lower()
            count += 1
            if word in vocab:  # whether to judge if in vocab
                vector = []
                for count, val in enumerate(values):
                    if count == 0:
                        continue
                    vector.append(float(val))
                word_vecs[word] = vector
    return word_vecs


def load_model_state(model, model_path):
    if os.path.exists(model_path):
        t0 = time.clock()
        logger.info("loading net_params ...")
        model.load_state_dict(torch.load(model_path))
        logger.info('load net_params taks %s' % (time.clock()-t0))
    return model


def main(args):
    text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    # text_field = data.Field(sequential=True, lower=True)
    label_field = data.Field(sequential=False)
    # label_field = data.Field(sequential=False, use_vocab=False)
    train_iter, dev_iter = data_loader.load_mr(
        text_field, label_field, batch_size=16)
    # text_field.vocab.load_vectors(wv_type='glove.6B', wv_dim=100)
    args.class_num = 2
    args.embed_num = len(text_field.vocab)
    args.embed_dim = 100
    args.hidden_dim = 50
    args.batch_size = 16

    model = LSTM(args)
    try:
        model = load_model_state(model, "./data/net_params.pkl")
        # model.word_embeddings.weight.data.copy_(text_field.vocab.vectors)
        # model.word_embeddings.weight.data = text_field.vocab.vectors
    except Exception as e:
        logger.error("load model fail: %s" % e, exc_info=True)
    else:
        best_dev = train.train_epoch(model, dev_iter, False)
        logger.info("Now dev acc: %s" % best_dev)
    logger.info("Train ....")
    t0 = time.time()
    train.train(model, train_iter, dev_iter, best_dev)
    logger.info("Train END! TK: %s And Test start" % (time.time() - t0))

    logger.info("loading test ...")
    results = {}
    test = pd.read_csv('./data/testData.tsv', delimiter="\t")
    for i in tqdm(range(len(test))):
        sentence = test['review'][i]
        # sentence = sentence.split(" ")
        label = train.predict(sentence, model, text_field, label_field)
        label = 0 if label == 'neg' else 1
        results.setdefault('id', []).append(test['id'][i])
        results.setdefault('sentiment', []).append(label)
        print (test['id'][i], label)
    submission = pd.DataFrame({'id': results['id'],
                               'sentiment': results['sentiment']})
    submission.to_csv('./data/lstm_submission.csv', index=False, sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
