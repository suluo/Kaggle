#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : test.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-02-26
# Last Modified: 2018-02-26 17:03:10
# Descption    :
# Version      : Python 2.7
############################################
from __future__ import division
import logging
import logging.handlers
# import traceback
# import os
import argparse
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


# file
logging.basicConfig(
    format="[ %(levelname)1.1s  %(asctime)s  %(module)s:%(lineno)d  %(name)s  ]  %(message)s",
    datefmt="%y%m%d %H:%M:%S",
    filemodel="a",
    filename="./data_dump.log",
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
logging.getLogger("").addHandler(console)

# Now, define a couple of other loggers which might represent areas in your
# application:
logger = logging.getLogger(__file__)


def restore_net():
    net = torch.load('./data/net.pkl')
    return net

def predict():


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='demo input num')
    args = parser.parse_args()
    main(args.num)


