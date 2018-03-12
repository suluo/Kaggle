#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : lstm.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-11 13:02:52
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import torch
import torch.nn.functional as F
from torch import autograd, nn
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        )

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)
