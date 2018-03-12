#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    :
# Author       : Suluo - sampson.suluo@gmail.com
# Creation Date: 2017-08-05
# Last Modified: 2017-08-05 01:24:42
# Descption    :
############################################
from __future__ import division
import logging
import logging.handlers
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from base_model import model_quality, select_params
from proprecess.clear import select_features

def select_model(x_train, y_trian):
    pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
    pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
    print 'count-mnb: ', model_quality(pip_count, x_train, y_train)
    print 'tfidf-mnb: ', model_quality(pip_tfidf, x_train, y_train)

def main():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    x_train, y_train, x_test = select_features(train, test)
    select_model(x_train, y_train)
    return 0

    params_count = {
        'count_vec_binary': [True, False],
        'count_vec_ngram_range': [(1, 1), (1, 2)],
        'mnb_alpha': [0.1, 1.0, 10.0]
    }
    params_tfidf = {
        'tfidf_vec_binary': [True, False],
        'tfidf_vec_ngram_range': [(1, 1), (1, 2)],
        'mnb_alpha': [0.1, 1.0, 10.0]
    }
    gs = select_params(pip_tfidf, params_tfidf, x_train, y_train)
    gs.fit(x_train, y_train)
    best_y_predict = gs.predict(x_test)
    best_submission = pd.DataFrame({'id': test['id'], 'sentiment': best_y_predict})
    best_submission.to_csv('../result/best_submission.csv', index=False)

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


