#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : lstm.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-04-04 16:49:02
# Descption    :
# Version      : Python 3.6
############################################
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.num_layers = 1
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size if 'batch_size' in args else 1

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        dropout = args.dropout if 'dropout' in args else 0.5

        # self.embed = nn.Embedding(V, D, max_norm=args.max_norm)
        self.embedding = nn.Embedding(V, D)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=dropout, num_layers=self.num_layers)
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (
            Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
            Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        )

    def forward(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence)
        # x = embeds.view(len(sentence), self.batch_size, -1)
        x = embeds.view(len(sentence), embeds.size(1), -1)
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
