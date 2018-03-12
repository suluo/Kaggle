#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    :
# Author       : Suluo - sampson.suluo@gmail.com
# Creation Date: 2017-08-05
# Last Modified: 2017-08-05 00:45:44
# Descption    :
############################################
from __future__ import division
import logging
import logging.handlers
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from bs4 import BeautifulSoup
import re

from nltk.corpus import stopwords

def review_to_text(review, remove_stopwords=True):
    raw_text = BeautifulSoup(review, 'html').get_text()
    lettters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words


def propress(train, test):
    x_train = []
    for review in train['review']:
        x_train.append(' '.join(review_to_text(review, True)))
    x_test = []
    for review in test['review']:
        x_test.append(' '.join(review_to_text(review, True)))
    y_train = train['sentiment']
    return x_train, y_train, x_test


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


