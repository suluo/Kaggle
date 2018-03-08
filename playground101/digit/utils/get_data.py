#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : getdata.py
# Purpose:
# Creation Date: 2017-07-14
# Last Modified: 2018-02-26 18:14:31
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division
import sys

from torchvision import datasets, transforms


def get_data(status=0):

    '''
    torchvision数据集的输出是在[0, 1]范围内的PILImage图片。
    我们此处使用归一化的方法将其转化为Tensor，数据范围为[-1, 1]
    '''
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    train_dataset = datasets.MNIST(
        root='./train_data', train=True, transform=data_tf, download=True
    )
    test_dataset = datasets.MNIST(
        root='./test_data', train=False, transform=data_tf, download=True
    )
    '''
    if status == 'train':
        return train_dataset
    elif status == 'test':
        return test_dataset
    else:
        return train_dataset, test_dataset

    '''
    return train_dataset, test_dataset


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
