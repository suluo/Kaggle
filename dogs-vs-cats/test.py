#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : test.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-08 11:17:17
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
import sys
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)

def test():
    correct = 0
    total = 0

    for data in test_loader:
        images,labels = data
        outputs = net(Variable(images))
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 5000 test images: %d %%' % (100 * correct / total))


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


