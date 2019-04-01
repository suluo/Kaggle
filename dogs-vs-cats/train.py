#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : train.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-08 11:20:35
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
from tqdm import tqdm
from model.base_net import Net
from util import dataset
import torch
from torch import nn, optim
from torch.autograd import Variable

# import os
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


def train():
    net = Net()
    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = 0.0001,momentum = 0.9)

    train_loader, test_loader = dataset.pro_progress()
    for epoch in tqdm(range(10)):
        running_loss = 0.0

        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            inputs,labels = Variable(inputs),Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = cirterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 2000 == 1999:
                print('[%d %5d] loss: %.3f' % (epoch + 1,i + 1,running_loss / 2000))
                running_loss = 0.0

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
    train()
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


