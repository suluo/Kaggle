#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : train.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-14 20:58:25
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchtext import data
from tqdm import tqdm
import random
import os
import logging
import logging.config
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
from model.lstm import LSTM
from proprecess.data_loader import dataset
import proprecess.data_loader_batch as data_loader
import spacy
spacy_en = spacy.load('en')
#logging.config.fileConfig('../conf/logging.conf')
#logger = logging.getLogger(__file__)
dataset = dataset()


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def train():
    # train_loader, dev_loader = dataset.load_train(batch_size=16)
    # word_vocab, label_vocab = dataset.load_vocab()
    # model = LSTM(embedding_dim=150, hidden_dim=150,
    #              vocab_size=len(word_vocab),
    #              label_size=len(label_vocab))
    text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    label_field = data.Field(sequential=False, use_vocab=False)
    train_loader, dev_loader = data_loader.load_mr(text_field, label_field, batch_size=16)
    model = LSTM(embedding_dim=100, hidden_dim=50,
                 vocab_size=len(text_field.vocab),
                 label_size=2)
    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=le-2)

    best_dev_acc = 0.0
    for i in range(10):
        print('epoch: %d start...' % i)
        train_epoch(model, train_loader, loss, optimizer, i)
        #logger.info('now best dev acc:', best_dev_acc)
        dev_acc = evaluate(model, dev_loader, loss)
        # test_acc = evaluate(model, test_data, loss, word_to_ix, label_to_ix, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            # os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(),
                       './data/state'+str(int(best_dev_acc*1000))+'.model')
        print('now best dev acc:', best_dev_acc)


def evaluate(model, loader, loss_function):
    model.eval()
    total, correct, avg_loss = 0, 0.0, 0.0

    for i, (sents, labels) in tqdm(loader):
        sents = Variable(sents)
        labels = Variable(labels)

        outputs = model(sents)
        _, predicted = torch.max(outputs.data, 1)

        loss = loss_function(outputs, labels)
        avg_loss = loss.data[0]
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # model.zero_grad() # should I keep this when I am evaluating the model?
    print('Evaluate - loss: {:.6f} acc: {:.4f}'.format(avg_loss/total, correct/total))
    return correct/total


def test(model, loader, word_to_ix):
    model.eval()

    labels = []
    for sent in tqdm(loader):
        # detaching it from its history on the last instance.
        sent = dataset.prepare_sequence(sent, word_to_ix)
        sent = Variable(sent)
        outputs = model(sent)
        _, label = torch.max(outputs.data, 1)
        labels.append(label)
    return labels


def train_epoch(model, train_loader, loss_function, optimizer, epoch):
    model.train()
    count = 0

    for step, (sents, labels) in tqdm(train_loader):
        sents = Variable(sents)
        labels = Variable(labels)
        # detaching it from its history on the last instance.
        model.batch_size = len(labels.size(0))
        model.hidden = model.init_hidden()

        # forward + backward + optimize
        optimizer.zero_grad()
        model.zero_grad()

        outputs = model(sents)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        count += 1
        if count % 1000 == 0:
            print('epoch: %d iterations: %d loss :%g' % (epoch, count*model.batch_size, loss.data[0]))


def main(num):
    train()
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


