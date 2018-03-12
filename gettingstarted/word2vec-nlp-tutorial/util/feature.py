#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    :
# Author       : Suluo - sampson.suluo@gmail.com
# Creation Date: 2017-08-05
# Last Modified: 2017-08-05 01:52:44
# Descption    :
############################################
from __future__ import division
import logging
import logging.handlers
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np

# 词向量产生文本特征向量
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    index2word_set = set(model.index2word)
    nwords = 0
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

# 影评转化成特征向量
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs

clean_train_review = []
for review in train['review']:
    clean_train_reviews.append(review_to_text(review, True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
clean_test_review = []
for review in train['review']:
    clean_test_reviews.append(review_to_text(review, True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


