#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name: features_selected.py
# Purpose:
# Creation Date: 2017-07-31
# Last Modified: 2017-08-16 17:54:46
# Actor by: Suluo - sampson.suluo@gmail.com
############################################
from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


def select_features(train, test):
    # features
    y_train = train['SalePrice']
    # 人工
    x_train = train.drop(['Id', 'SalePrice'], axis=1)
    x_test = test.drop(['Id'], axis=1)

    # 数据填充
    x_train['LotFrontage'].fillna(x_train['LotFrontage'].mean(), inplace=True)
    x_train['MasVnrArea'].fillna(x_train['MasVnrArea'].mean(), inplace=True)
    x_train['GarageYrBlt'].fillna(x_train['GarageYrBlt'].mean(), inplace=True)
    x_train['MasVnrArea'].fillna(x_train['MasVnrArea'].mean(), inplace=True)

    x_test['LotFrontage'].fillna(x_test['LotFrontage'].mean(), inplace=True)
    x_test['MasVnrArea'].fillna(x_test['MasVnrArea'].mean(), inplace=True)
    x_test['BsmtFinSF1'].fillna(x_test['BsmtFinSF1'].mean(), inplace=True)
    x_test['BsmtFinSF2'].fillna(x_test['BsmtFinSF2'].mean(), inplace=True)
    x_test['BsmtUnfSF'].fillna(x_test['BsmtUnfSF'].mean(), inplace=True)
    x_test['TotalBsmtSF'].fillna(x_test['TotalBsmtSF'].mean(), inplace=True)
    x_test['BsmtFullBath'].fillna(x_test['BsmtFullBath'].mean(), inplace=True)
    x_test['BsmtHalfBath'].fillna(x_test['BsmtHalfBath'].mean(), inplace=True)
    x_test['GarageYrBlt'].fillna(x_test['GarageYrBlt'].mean(), inplace=True)
    x_test['GarageCars'].fillna(x_test['GarageCars'].mean(), inplace=True)
    x_test['GarageArea'].fillna(x_test['GarageArea'].mean(), inplace=True)
    x_train.fillna('UNKNOWN', inplace=True)
    x_test.fillna('UNKNOWN', inplace=True)
    # x_train = x_train.dropna(how='any')

    print x_train.info(), x_test.info()

    dict_vec = DictVectorizer(sparse=False)
    x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
    # print "feature_names:", dict_vec.feature_names_
    x_test = dict_vec.transform(x_test.to_dict(orient='record'))

    from sklearn import preprocessing
    # normalize the data attributes
    # x_train = preprocessing.normalize(x_train)
    # x_test = preprocessing.normalize(x_test)
    # standardize the data attributes
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)
    return x_train, y_train, x_test


def main(argv):
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv("../data/test.csv")
    x_train, y_train, x_test = select_features(train, test)
    pass

if __name__ == "__main__":
    main(sys.argv)


