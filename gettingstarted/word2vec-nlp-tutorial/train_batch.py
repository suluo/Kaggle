#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : train_batch.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-15 13:35:13
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
import logging
import logging.config
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
from model.lstm import LSTM
# from proprecess import data_loader
import proprecess.data_loader_batch as data_loader

# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)
import spacy
spacy_en = spacy.load("en")


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def train():
    # text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    text_field = data.Field(sequential=True, lower=True)
    label_field = data.Field(sequential=False)
    # label_field = data.Field(sequential=False, use_vocab=False)
    train_iter, dev_iter = data_loader.load_mr(
        text_field, label_field, batch_size=16)
    # text_field.vocab.load_vectors(wv_type='glove.6B', wv_dim=100)
    model = LSTM(embedding_dim=100, hidden_dim=50,
                 vocab_size=len(text_field.vocab),
                 label_size=2)
    # model.word_embeddings.weight.data = text_field.vocab.vectors
    if torch.cuda.is_available():
        model = model.cuda()

    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=le-2)

    best_dev_acc = 0.0
    for i in range(10):
        print('epoch: %d start...' % i)
        train_epoch(model, train_iter, loss, optimizer, text_field, label_field, i)
        # logger.info('now best dev acc:', best_dev_acc)
        dev_acc = evaluate(model, dev_iter, loss, 'dev')
        # test_acc = evaluate(model, test_data, loss, word_to_ix, label_to_ix, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            # os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(),
                       './data/state'+str(int(best_dev_acc*1000))+'.model')
        print('now best dev acc:', best_dev_acc)


def evaluate(model, eval_iter, loss_function, name='dev'):
    model.eval()
    avg_loss, corrects = 0.0, 0.0

    for batch in tqdm(eval_iter):
        sents, labels = batch.text, batch.label.data.sub_(1)
        # feature, target = Variable(sents), Variable(labels)
        feature, target = sents, Variable(labels)
        if torch.cuda.is_available():
            feature, target = feature.cuda(), target.cuda()

        model.batch_size = batch.batch_size
        model.hidden = model.init_hidden()

        logit = model(feature)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(logit, target)

        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        avg_loss += loss.data[0]
    eval_size = len(eval_iter.dataset)
    avg_loss /= eval_size
    print('eval avg_loss: {} , acc: {}'.format(avg_loss/eval_size, corrects/eval_size))
    return corrects/eval_size


def train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i):
    model.train()
    avg_loss = 0.0
    count = 0

    for batch in tqdm(train_iter):
        sents, labels = batch.text, batch.label.data.sub_(1)
        # feature, target = Variable(sents), Variable(labels)
        feature, target = sents, Variable(labels)
        if torch.cuda.is_available():
            feature, target = feature.cuda(), target.cuda()

        model.batch_size = batch.batch_size
        model.hidden = model.init_hidden()

        optimizer.zero_grad()
        model.zero_grad()
        logit = model(feature)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(logit, target)
        loss.backward()
        optimizer.step()

        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = float(corrects)/batch.batch_size
        avg_loss += loss.data[0]
        count += 1
        if count % 200 == 0:
            print('epoch: {} iterations: {} loss: {}'.format(i, count*model.batch_size, loss.data[0]))
    train_size = len(train_iter.dataset)
    avg_loss /= train_size
    print('epoch: {} done! train avg_loss: {}, acc: {}'.format(i, avg_loss, accuracy))


def main(num):
    train()
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)
