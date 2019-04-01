#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : feature_net.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-07
# Last Modified: 2018-03-07 16:03:30
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
# import os
import sys
from torch import nn
from torchvision import models
import logging
import logging.handlers
# import traceback
# file
logging.basicConfig(
    format="[ %(levelname)1.1s  %(asctime)s  %(module)s:%(lineno)d  %(name)s  ]  %(message)s",
    datefmt="%y%m%d %H:%M:%S",
    filemode="a",
    filename="./log/test.log",
    stream=sys.stdout, # 默认stderr, 和filename同时指定时，stream被忽略
    level=logging.INFO
)

# console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
format="[ %(levelname)1.1s  %(asctime)s  %(module)s:%(lineno)d  %(name)s  ]  %(message)s",
formatter = logging.Formatter(format)
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
# logging.getLogger(__name__).addHandler(console)
logging.getLogger().addHandler(console)

# Now, define a couple of other loggers which might represent areas in your
# application:
logger = logging.getLogger(__file__)


class feature_net(nn.Module):
    def __init__(self, model):
        super(feature_net, self).__init__()
        if model == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.feature == nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(9))
        elif model == 'inceptionv3':
            inception = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inception.children())[:-1])
            self.feature._modules.pop('13')
            self.feature.add_modules('global average', nn.AvgPool2d(35))
        elif model == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """
        model includes vgg19, inceptionv3, resnet152
        """
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x


class classifier(nn.Module):
    def __init(self, dim, n_classes):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, n_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


