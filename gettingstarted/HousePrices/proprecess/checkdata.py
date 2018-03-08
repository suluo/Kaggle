#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name: checkdata.py
# Purpose:
#           xgboost :http://blog.csdn.net/u014365862/article/details/73739857
# Creation Date: 2017-07-30
# Last Modified: 2017-08-14 20:41:53
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division
import logging
import logging.handlers
#import traceback
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
#查看数据信息
print '********train********'
print train.info()
print '**********test*******'
print test.info()

print train['MasVnrType'].value_counts()
print train['BsmtQual'].value_counts()
print train['BsmtCond'].value_counts()
print train['BsmtExposure'].value_counts()
print train['BsmtFinType1'].value_counts()
print train['BsmtFinType2'].value_counts()
print train['Electrical'].value_counts()
# LotFrontage, MasVnrArea
x_columns = []
x_train = train[x_columns]
del_columns = ['Id', 'MSZoning', 'Street', 'Alley']
train.drop(del_columns, axis=1)

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


