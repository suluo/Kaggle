#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    :
# Author       : Suluo - sampson.suluo@gmail.com
# Creation Date: 2017-08-05
# Last Modified: 2017-08-05 00:08:13
# Descption    :
############################################
from __future__ import division
import logging
import logging.handlers
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd

train = pd.read_csv('../data/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../data/testData.tsv', delimiter='\t')

train.head()
test.head()


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


