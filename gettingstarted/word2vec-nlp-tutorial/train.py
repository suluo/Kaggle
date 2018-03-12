#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : train.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-12 09:26:34
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import torch
from torch import nn, optim
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

#logging.config.fileConfig('../conf/logging.conf')
#logger = logging.getLogger(__file__)
dataset = dataset()


def train():
    train_data, dev_data, word_to_ix, label_to_ix = \
        dataset.load_train()
    model = LSTM(embedding_dim=150, hidden_dim=150,
                 vocab_size=len(word_to_ix),
                 label_size=len(label_to_ix))
    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=le-2)

    best_dev_acc = 0.0
    for i in range(10):
        random.shuffle(train_data)
        print('epoch: %d start...' % i)
        train_epoch(model, train_data, loss, optimizer, word_to_ix, label_to_ix, i)
        #logger.info('now best dev acc:', best_dev_acc)
        dev_acc = evaluate(model, dev_data, loss, word_to_ix, label_to_ix, 'dev')
        # test_acc = evaluate(model, test_data, loss, word_to_ix, label_to_ix, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            # os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(),
                       './data/state'+str(int(best_dev_acc*1000))+'.model')
        print('now best dev acc:', best_dev_acc)


def get_accuracy(truth, pred):
     assert len(truth) == len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i] == pred[i]:
             right += 1.0
     return right/len(truth)


def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in tqdm(data):
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = dataset.prepare_sequence(sent, word_to_ix)
        label = dataset.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc))
    return acc


def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in tqdm(train_data):
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = dataset.prepare_sequence(sent, word_to_ix)
        label = dataset.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 1000 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))



def main(num):
    train()
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


