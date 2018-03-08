#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : train.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-02-26
# Last Modified: 2018-03-06 17:57:47
# Descption    :
# Version      : Python 2.7
############################################
from __future__ import division
import logging
import logging.handlers
# import traceback
# import os
import argparse


# file
logging.basicConfig(
    format="[ %(levelname)1.1s  %(asctime)s  %(module)s:%(lineno)d  %(name)s  ]  %(message)s",
    datefmt="%y%m%d %H:%M:%S",
    filemode="w",
    filename="./log/test.log",
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

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net.simpleNet import Net
from net.cnn import CNN
from net.rnn import RNN
from utils import getdata


def get_data():
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_data = datasets.CIFAR10(
        root="./train_data", train=True, transform=data_tf, download=True
    )
    test_data = datasets.CIFAR10(
        root="./test_data", train=False, transform=data_tf, download=True
    )
    return train_data, test_data


def train():
    '''

    '''
    try:
        # train_data = getdata.get_data('train')
        # test_data = getdata.get_data('test')
        train_data, test_data = getdata.get_data()
        trainloader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            num_workers=2
        )
        testloader = DataLoader(
            test_data,
            batch_size=64,
            shuffle=True,
            num_workers=2
        )
    except Exception as e:
        logger.error("Error in get_data: %s" % e, exc_info=True)
    # logger.info('Success in Get_data')

    # model = Net(28*28, 300, 100, 10)
    model = CNN()
    #model = RNN()

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    print ("Train...")
    sumloss, best_acc = 0.0, 0.0
    for epoch in range(10):
        for step, (x, y) in enumerate(trainloader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            b_x = Variable(x)
            b_y = Variable(y)
            output = model(b_x)
            loss = criterion(output, b_y)
            sumloss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
            #       batch_x.numpy(), '| batch y: ', batch_y.numpy())

        # model.eval()
        eval_loss, eval_acc = 0, 0
        for data in testloader:
            img, label = data
            #img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.data[0]

        print('Epoch: [{}/10], SumLoss: {:.6f}, Loss: {:.6f}, ACC: {:.6f}'.format(
            epoch, sumloss, eval_loss/len(testloader), eval_acc/len(test_data)
        ))
        if eval_acc/len(testloader) > best_acc:
            torch.save(model, './data/model.pkl')
            best_acc = eval_acc/len(testloader)



def main(num):
    train()
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='demo input num')
    args = parser.parse_args()
    main(args.num)
