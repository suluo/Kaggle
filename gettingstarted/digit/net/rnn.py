#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : rnn.py
# Purpose:
# Creation Date: 2017-07-14
# Last Modified: 2018-03-06 16:49:03
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division
import sys

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        r_out, (h_n, h_c) = self.lstm(x, None)
        # choose r_out at the last time step
        out = self.classifier(r_out[:, -1, :])
        return out


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
