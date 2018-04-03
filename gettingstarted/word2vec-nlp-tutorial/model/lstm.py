#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : lstm.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-29 18:07:13
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = 1
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, int(hidden_dim/2), self.num_layers, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (
            Variable(torch.zeros(self.num_layers, self.batch_size, int(self.hidden_dim/2))),
            Variable(torch.zeros(self.num_layers, self.batch_size, int(self.hidden_dim/2)))
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
