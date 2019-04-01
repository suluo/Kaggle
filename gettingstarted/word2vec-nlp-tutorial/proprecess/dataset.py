#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : dataset.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-13
# Last Modified: 2018-03-13 19:19:26
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
import sys
import torch.utils.data as data
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)


class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# Then, you can just use prebuilt torch's data loader.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)



def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)


