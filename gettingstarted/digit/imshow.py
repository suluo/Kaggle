#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name: imshow.py
# Purpose:
# Creation Date: 2017-07-14
# Last Modified: 2017-07-14 16:41:31
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

# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

# show some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))


def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)


