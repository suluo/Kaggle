#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name: features_selected.py
# Purpose:
# Creation Date: 2017-07-31
# Last Modified: 2017-08-08 14:02:36
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
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import feature_selection

def select_features(train, test):
    # features
    y_train = train['Survived']
    # 人工
    selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
    x_train = train[selected_features]
    x_test = test[selected_features]
    # x_train.drop(['row.names', 'name', 'survived'], axis=1)

    # 数据填充
    x_train['Embarked'].fillna('S', inplace=True)
    x_test['Embarked'].fillna('S', inplace=True)
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
    x_train['Fare'].fillna(x_train['Fare'].mean(), inplace=True)
    x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
    #x_train = x_train.dropna(how='any')

    print x_train.info(), x_test.info(), y_train.value_counts()

    dict_vec = DictVectorizer(sparse=False)
    x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
    print "feature_names:", dict_vec.feature_names_
    x_test = dict_vec.transform(x_test.to_dict(orient='record'))

    return x_train, y_train, x_test


def main(argv):
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv("../data/test.csv")
    pass

if __name__ == "__main__":
    main(sys.argv)


