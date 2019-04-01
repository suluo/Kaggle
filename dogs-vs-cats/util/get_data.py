#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : dataset.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-07
# Last Modified: 2018-03-08 10:44:03
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
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


def pro_progess(filepath="../data"):
    height = 299
    train_files = os.listdir(filepath + '/train')
    train = np.zeros((len(train_files), height, height, 3), dtype=np.uint8)
    labels = list(filter(lambda x: x[:3] == 'dog', train_files))

    test_files = os.listdir(filepath + '/test')
    test = np.zeros((len(test_files), height, height, 3), dtype=np.uint8)

    for i in tqdm(range(len(train_files))):
        filename = filepath + train_files[i]
        img = cv2.imread(filename)
        img = cv2.resize(img, (height, height))
        train[i] = img[:, :, ::-1]

    for i in tqdm(range(len(test_files))):
        filename = filepath + test_files[i]
        img = cv2.imread(filename)
        img = cv2.resize(img, (height, height))
        test[i] = img[:, :, ::-1]

    print ('Training Data Size = %.2 GB' % (sys.getsizeof(train)/1024**3))
    print ('Testing Data Size = %.2 GB' % (sys.getsizeof(test)/1024**3))
    X_train, X_val, y_train, y_val = train_test_split(
        train, labels, shuffle=True, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


def pro_progress():
    data_transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
    ])

    train_dataset = datasets.ImageFolder(root = '../data/train/',transform = data_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4)

    test_dataset = datasets.ImageFolder(root = '../data/test/',transform = data_transform)
    test_loader = DataLoader(
        test_dataset, batch_size = 4, shuffle = True, num_workers = 4)
    return train_loader, test_loader



def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


