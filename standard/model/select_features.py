#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name: select_features.py
# Purpose:
# Creation Date: 2017-07-31
# Last Modified: 2017-08-02 01:56:40
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

def select_features():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    # features
    y_train = train['Survived']
    # 人工
    # selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
    # x_train = train[selected_features]
    # x_test = test[selected_features]
    # x_columns = [x for x in train.columns if x not in ['row.names', 'name', 'survived']]
    # x_train = train[x_columns]
    x_train.drop(['row.names', 'name', 'survived'], axis=1)

    # 数据填充
    x_train['Embarked'].fillna('S', inplace=True)
    x_test['Embarked'].fillna('S', inplace=True)
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
    x_train['Fare'].fillna(x_train['Fare'].mean(), inplace=True)
    #x_train = x_train.dropna(how='any')

    x_train.fillna('UNKNOWN', inplace=True)
    # 决策树选择特征 : 前20%
    # for i in range(1, 100, 2): # 最佳性能特征筛选
    fs = feature_selection.SelectPrecentile(feature_selection.chi2, percentile=20)
    x_train_fs = fs.fit_transform(x_train, y_train)
    # scores = cross_val_score(model, x_train_fs, y_train, cv=5)
    # results = np.append(results, scores.mean())
    # opt = np.where(results == results.max())[0]

    dict_vec = DictVectorizer(sparse=False)
    x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
    print dict_vec.feature_names_
    x_test = dict_vec.transform(x_test.to_dict(orient='record'))
    # 标准化
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    return x_train, y_train, x_test

# 文本特征向量转化
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# vec = CountVectorizer()
# vec = TfidfVectorizer()
# 去掉停用词
vec = CountVectorizer(analyzer='word', stop_words='english')
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
print vec.get_feature_names()

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


