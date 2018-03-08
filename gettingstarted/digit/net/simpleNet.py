#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : simpleNet.py
# Purpose:
# Creation Date: 2017-07-14
# Last Modified: 2018-02-26 18:09:56
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
