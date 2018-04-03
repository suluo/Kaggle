#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : train_batch.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-04-03 12:10:28
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import torch
from torch import nn, optim
from tqdm import tqdm
from torchtext import data
from torch.autograd import Variable
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


def train():
    text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    # text_field = data.Field(sequential=True, lower=True)
    label_field = data.Field(sequential=False)
    # label_field = data.Field(sequential=False, use_vocab=False)
    train_iter, dev_iter = data_loader.load_mr(
        text_field, label_field, batch_size=16)
    # text_field.vocab.load_vectors(wv_type='glove.6B', wv_dim=100)
    model = LSTM(embedding_dim=100, hidden_dim=50,
                 vocab_size=len(text_field.vocab),
                 label_size=2)

    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2)
    if torch.cuda.is_available():
        model = model.cuda()

    try:
        model = load_model_state(model, "./data/net_params.pkl")
        # model.word_embeddings.weight.data.copy_(text_field.vocab.vectors)
        # model.word_embeddings.weight.data = text_field.vocab.vectors
    except Exception as e:
        logger.error("load model fail: %s" % e, exc_info=True)
    else:
        best_dev_acc = train_epoch(model, dev_iter, loss, optimizer, False)

    for i in range(10):
        t0 = time.time()
        logger.info('Epoch : %s start ...' % i)
        train_epoch(model, train_iter, loss, optimizer, epoch=i)
        dev_acc = train_epoch(model, dev_iter, loss, optimizer, False)
        # test_acc = evaluate(model, test_data, loss, word_to_ix, label_to_ix, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print('New Best Dev - {} !!!'.format(best_dev_acc))
            torch.save(model.state_dict(), "./data/net_params.pkl")
        logger.info('epoch %s taks %s h, now best dev acc %s'
                    % (i, (time.time()-t0)/3600, best_dev_acc))


def train_epoch(model, train_iter, loss_func, optimizer, is_train=True, epoch=0):
    if is_train:
        model.train()
    else:
        model.eval()
    avg_loss, corrects, count = 0.0, 0.0, 0

    for batch in tqdm(train_iter):
        sents, labels = batch.text, batch.label.data.sub_(1)
        # feature, target = Variable(sents), Variable(labels)
        feature, target = sents, Variable(labels)
        if torch.cuda.is_available():
            feature, target = feature.cuda(), target.cuda()

        model.batch_size = batch.batch_size
        model.hidden = model.init_hidden()

        logit = model(feature)
        loss = loss_func(logit, target)

        batch_correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        avg_loss += loss.data[0]
        if is_train:
            # model.zero_grad() # should I keep this when I am evaluating the model?
            # optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 200 == 0:
                # print('train-epoch: {} iterations: {} loss: {}'.format(epoch, count*model.batch_size, loss.data[0]))
                logger.info('train-epoch: %s iterations: %s loss: %s acc: %s'
                            % (epoch, count*model.batch_size, loss.data[0], float(batch_correct)/model.batch_size))

    train_size = len(train_iter.dataset)
    avg_loss /= train_size
    accuracy = float(corrects)/train_size
    logger.info('Is_train: %s epoch: %s avg_loss: %s, acc: %s'
                % (is_train, epoch, avg_loss, accuracy))
    return accuracy


def predict(model, sent):
    to_ix = {}
    model = model.eval()
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        tensor = torch.LongTensor(idxs)
        return Variable(tensor)
    sentence = prepare_sequence(sent, to_ix)
    logit = model(sentence)
    return torch.max(logit, 1)[1].data


def main(num):
    train()
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)
