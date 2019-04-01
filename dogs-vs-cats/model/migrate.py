#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : migrate.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-07
# Last Modified: 2018-03-07 14:41:19
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
from torchvision import models
import torch.nn as nn
import sys
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


transfer_model = models.resnet18(pretrained=True)
dim_in = transfer_model.fc.in_features
transfer_model.fc = nn.Linear(dim_in, img_classes)

def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


