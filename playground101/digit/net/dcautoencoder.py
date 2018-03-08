#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : dcautoencoder.py
# Purpose:
# Creation Date: 2017-07-14
# Last Modified: 2018-03-01 20:34:33
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division
import sys

import torch.nn as nn


class DCautoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class, n_layer=2):
        super(DCautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
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
