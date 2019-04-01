#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : model_selected.py
# Creation Date: 2017-08-01
# Last Modified: 2018-04-16 15:44:57
# Actor by: Suluo - sampson.suluo@gmail.com
# Purpose:
# scikit-learn的主要模块和基本使用:http://blog.csdn.net/u013066730/article/details/54314136
# scikit-learn 详细文档：http://scikit-learn.org/stable/user_guide.html
# grid_search参数选择：http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
############################################
from __future__ import division
import logging
import logging.handlers
#import traceback
import os
import sys
import pandas as pd
reload(sys)
sys.setdefaultencoding("utf-8")
import select_features

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
gbc = GradientBoostingClassifier(random_state=10)
# 回归
gbr = GradientBoostingRegressor()
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
rfc = RandomForestClassifier()
# 回归
etr = ExtraTreesRegressor()
rfr = RandomForestRegressor()
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
svc_params = {'svc_gamma': np.logspace(-2, 1, 4), 'svc_C': np.logspace(-1, 1, 3)}
# 回归
svr = SVR()
svr_params = {'kernel': ['linear', 'poly', 'rbf']}
from sklearn.naive_bayes import MultinomialNB, GaussianNB
mnb = MultinomialNB()
gnb = GaussianNB()
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
knn = KNeighborsClassifier()
# 回归
uni_knr = KNeighborsRegressor(weights='uniform')
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
dtc = DecisionTreeClassifier()
# 回归
dtr = DecisionTreeRegressor()

# 聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
from sklearn.decomposition import PCA
estimator = PCA(n_component=2) # 高维压缩到2维
# x_pca=estimator.fit_transform(x_digits)

from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
# 聚类
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
def model_quality(model, X_train, Y_train=None):
    # 数据分割
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
    # model.fit(x_train, y_train)
    # model_y_predict = model.predict(x_test)
    # model.score(X_test, y_test)
    # metrics.accuracy_score(y, y_pred)
    # 详细分类性能
    # model_y_predict = model.predict(x_test)
    # classification_report(y_test, model_y_predict, target_names=[])
    # 回归性能评估
    # r2_score(y_test, model_y_predict)
    # mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(model_y_predict))
    # mean_absolute_error(y_test, model_y_predict)
    # 特征贡献度
    # np.sort(zip(model.feature_importances_, boston.feature_names), axis=0)
    # 聚类性能评估
    # model.fit(X_train)
    # y_pred = model.predict(X_test)
    # metrics.adjusted_rand_score(y_test, y_pred)
    # 轮廓系数
    # silhouette_score(X, model.labels_, metric='euclidean')
    # 肘部 观察法预估聚类簇个数
    #for k in range(1, 10):
    #    kmeans = Kmeans(n_clusters=k)
    #    kmeans.fit(X)
    #    sum(np.min(cdist(X, keans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0]
    return cross_val_score(model, X_train, Y_train, cv=5).mean()


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
def select_params(model, params, x_train, y_train):
    # pipeline 简化系统搭建流程
    # clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
    # gs = GridSearchCV(clf, params, verbose=2, refit=True, cv=3)
    # n_jobs = -1全部cpu多线程，
    gs = GridSearchCV(model, params, scoring='roc_auc', n_jobs=-1, cv=5, verbose=2, refit=True)
    gs.fit(x_train, y_train)
    print gs.cv_results_
    print 'best_estimator', gs.best_estimator_
    print 'best_params', gs.best_params_
    print 'best_score', gs.best_score_
    return gs


def select_model(x_train, y_train):
    print "gbdt_score:", model_quality(gbc, x_train, y_train)
    print "rfc_score:", model_quality(rfc, x_train, y_train)
    print "lr_score:", model_quality(lr, x_train, y_train)
    print "svm_score:", model_quality(lsvc, x_train, y_train)


from sklearn.externals import joblib
def save_model(model):
    # lr是一个LogisticRegression模型
    joblib.dump(model, 'lr.model')
    return lr


def job_model(model_path):
    if not model_path:
        model_path = 'lr.model'
    model = joblib.load(model_path)
    return model


def main(argv):
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv('../data/test.csv')
    x_train, y_train, x_test = select_features.select_features(train, test)
    select_model(x_train, y_train)
    # gbdt
    #print x_train.info(), x_test.info(), y_train.value_counts()
    gbc_params = {
        "n_estimators": range(50, 1000, 50), # 默认100
        "learning_rate": [0.05, 0.1, 0.25, 0.5, 1.0], #默认0.1
        "max_features": range(7, 20, 2), #
        "subsample": [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9], # 默认1，合适0.5~0.8
        "max_depth": range(10, 100, 5),
        "min_samples_split": range(100, 1900, 200),
        "min_samples_leaf": range(60, 101, 10),
        "alpha": [0.7, 0.8, 0.9], # 默认0.9
    }
    #gbdt_params = {
    #    "subsample": [0.5, 0.6, 0.65, 0.7],
    #               }
    gs = select_params(gbc, gbc_params, x_train, y_train)
    save_model(gs.best_estimator_)
    return 0


if __name__ == "__main__":
    main(sys.argv)
