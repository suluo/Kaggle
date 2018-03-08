#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : autoencoder.py
# Purpose:
# Creation Date: 2017-07-14
# Last Modified: 2018-03-01 20:30:04
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division
import sys

import torch.nn as nn


class autoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class, n_layer=2):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
