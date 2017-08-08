#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    :
# Author       : Suluo - sampson.suluo@gmail.com
# Creation Date: 2017-08-08
# Last Modified: 2017-08-08 13:58:17
# Descption    :
############################################
from __future__ import division
import logging
import logging.handlers
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import model_selected
import features_selected

def main(argv):
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv('../data/test.csv')
    x_train, y_train, x_test = features_selected.select_features(train, test)
    # gbdt
    #print x_train.info(), x_test.info(), y_train.value_counts()
    gbdt_params = {
        "n_estimators": range(200, 600, 100),
        "learning_rate": [0.1, 0.2, 0.25, 0.3, 0.5],
        "max_features": range(8, 11),
        "subsample": [0.5, 0.55, 0.6, 0.65, 0.7],
        "max_depth": range(2, 6)
                   }
    gs = model_selected.select_params(gbdt, gbdt_params, x_train, y_train)
    gs.fit(x_train, y_train)
    best_y_predict = gs.predict(x_test)
    best_submission = pd.DataFrame({"PassengerId": test['PassengerId'],
                                    "Survived": best_y_predict})
    best_submission.to_csv("../result/gbdt_submission.csv", index=False)
    return 0


if __name__ == "__main__":
    main(sys.argv)


