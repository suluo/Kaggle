#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : data_loader_batch.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-14
# Last Modified: 2018-03-27 17:19:32
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
import sys
import logging
import logging.config
# logging.config.fileConfig('./conf/logging.conf')
# logger = logging.getLogger(__file__)
import re
import torch
from tqdm import tqdm
from torchtext import data
import pandas as pd
import random
seed_num = 1
torch.manual_seed(seed_num)
random.seed(seed_num)


class MR(data.Dataset):
    def __init__(self, text_field, label_field, path=None, file=None, examples=None, char_data=None, **kwargs):
        """
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            char_data: The char level to solve
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)

            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = None if os.path.join(path, file) is None else os.path.join(path, file)
            examples = []
            if path:
                print (67, 'loading {} start ....'.format(path))
                a, b = 0, 0
                train = pd.read_csv(path, delimiter="\t")
                for i in tqdm(range(len(train))):
                    sentence, flag = train['review'][i], train['sentiment'][i]
                    sentence = clean_str(sentence)
                    if char_data is True:
                        sentence = sentence.split(" ")
                        sentence = MR.char_data(self, sentence)
                    # print(sentence)
                    # clear string in every sentence
                    if flag == 0:
                        a += 1
                        examples += [data.Example.fromlist([sentence, 'neg'], fields=fields)]
                    elif flag == 1:
                        b += 1
                        examples += [data.Example.fromlist([sentence, "pos"], fields=fields)]
                print("a {} b {} ".format(a, b))
        super(MR, self).__init__(examples, fields, **kwargs)

    def char_data(self, list):
        data = []
        for i in range(len(list)):
            for j in range(len(list[i])):
                data += list[i][j]
        return data

    @classmethod
    def splits(cls, text_field, label_field, filename, char_data=False, dev_ratio=.1, shuffle=True, path='./data/', **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        print(path + filename)
        examples = cls(text_field, label_field, path=path, file=filename, char_data=char_data, **kwargs).examples
        if shuffle:
            print("shuffle data examples......")
            random.shuffle(examples)

        dev_index = -1 * int(dev_ratio*len(examples))
        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


def load_mr(text_field, label_field, batch_size, **kwargs):
        train_data, dev_data = MR.splits(text_field, label_field, filename='labeledTrainData.tsv')
        print("len(train_data) {}, len(dev_data): {} ".format(len(train_data), len(dev_data)))
        text_field.build_vocab(train_data.text, dev_data.text)
        # text_field.build_vocab(train_data, vectors="glove.6B.100d")
        label_field.build_vocab(train_data.label, dev_data.label)
        train_iter, dev_iter = data.Iterator.splits(
            (train_data, dev_data),
            batch_sizes=(batch_size, len(dev_data)),
            sort_key=lambda x: len(x.text),
            repeat=False, device=-1,
            **kwargs
        )
        print (136, len(train_iter), type(train_iter))
        return train_iter, dev_iter


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)
