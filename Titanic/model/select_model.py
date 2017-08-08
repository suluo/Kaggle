#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name: select_model.py
# Creation Date: 2017-08-01
# Last Modified: 2017-08-08 13:52:38
# Actor by: Suluo - sampson.suluo@gmail.com
# Purpose:
############################################
from __future__ import division
import logging
import logging.handlers
#import traceback
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import select_features

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(random_state=10)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
lr = LogisticRegression()
# from sklearn.preprocessing import PolynomialFeatures
# poly2 = PolynomialFeatures(degree=2)
# x_train_poly2.fit_transform(x_train)
lasso = Lasso()
lrrd = Ridge()
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier
from sklearn.svm import LinearSVC, SVC, SVR
lsvc = LinearSVC()
svc = SVC()

from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import train_test_split
# 聚类
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
def model_quality(model, X_train, Y_train=None):
    return cross_val_score(model, X_train, Y_train, cv=5).mean()

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
def select_params(model, params, x_train, y_train):
    # pipeline 简化系统搭建流程
    # clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
    # gs = GridSearchCV(clf, params, verbose=2, refit=True, cv=3)
    # n_jobs = -1全部cpu多线程，
    gs = GridSearchCV(model, params, n_jobs=-1, cv=5, verbose=1, refit=True)
    gs.fit(x_train, y_train)
    # print gs.grid_scores_
    print gs.best_score_, gs.best_params_
    return gs

def select_model(x_train, y_train):
    print "gbdt_score:", model_quality(gbdt, x_train, y_train)
    print "rfc_score:", model_quality(rfc, x_train, y_train)
    print "lr_score:", model_quality(lr, x_train, y_train)
    print "svm_score:", model_quality(lsvc, x_train, y_train)


def main(argv):
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv('../data/test.csv')
    x_train, y_train, x_test = select_features.select_features(train, test)
    select_model(x_train, y_train)
    # gbdt
    #print x_train.info(), x_test.info(), y_train.value_counts()
    gbdt_params = {
        "n_estimators": range(200, 600, 100),
        "learning_rate": [0.1, 0.2, 0.25, 0.3, 0.5],
        "max_features": range(8, 11),
        "subsample": [0.5, 0.55, 0.6, 0.65, 0.7],
        "max_depth": range(2, 6)
                   }
    #gbdt_params = {
    #    "subsample": [0.5, 0.6, 0.65, 0.7],
    #               }
    #gs = select_params(gbdt, gbdt_params, x_train, y_train)
    gs = GradientBoostingClassifier(max_features=10, max_depth=3, random_state=10)
    gs.fit(x_train, y_train)
    best_y_predict = gs.predict(x_test)
    best_submission = pd.DataFrame({"PassengerId": test['PassengerId'],
                                    "Survived": best_y_predict})
    best_submission.to_csv("../result/gbdt_submission.csv", index=False)

    pass

if __name__ == "__main__":
    main(sys.argv)


