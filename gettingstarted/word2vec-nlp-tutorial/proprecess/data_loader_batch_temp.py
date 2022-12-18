#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : data_loader_batch.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-03-14 11:19:51
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import os
import sys
import logging
import logging.config
# logging.config.fileConfig('../conf/logging.conf')
# logger = logging.getLogger(__file__)
import torch
import torch.autograd as autograd
import codecs
import pandas as pd
import random
from tqdm import tqdm
from torchtext import data
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import urllib
import re
SEED = 1


import re
import tarfile
import spacy
spacy_en = spacy.load('en')


class DataPro():

    @staticmethod
    def tokenizer(text): # create a tokenizer function
        # 返回 a list of <class 'spacy.tokens.token.Token'>
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def load_train(self):
        train_data = []
        train = pd.read_csv('./data/labeledTrainData.tsv', delimiter="\t")
        x_train = []
        print ("loading train data start...")
        for review in tqdm(train['review']):
            x_train.append(" ".join(self.review_to_text(review, False)))
        train_data = [(x, y) for x, y in zip(x_train, train['sentiment'])]
        return train_data

    def load_train_vocab(self):
        train = pd.read_csv('./data/labeledTrainData.tsv', delimiter="\t")
        x_train = []
        print ("loading train data start...")
        for review in tqdm(train['review']):
            x_train.append(" ".join(self.review_to_text(review, False)))

        train_data = [(x, y) for x, y in zip(x_train[:20000], train['sentiment'][:20000])]
        dev_data = [(x, y) for x, y in zip(x_train[20001:], train['sentiment'][20001:])]

        random.shuffle(train_data)
        random.shuffle(dev_data)

        word_to_ix = self.build_token_to_ix([s for s,_ in train_data + dev_data])
        label_to_ix = {0: 0, 1: 1}
        print('train:', len(train_data),
              'dev:',len(dev_data),
              'vocab size:',len(word_to_ix),
              'label size:',len(label_to_ix),
              'loading train data done!')
        return train_data, dev_data, word_to_ix, label_to_ix

    def load_unlabel_data(self):
        unlabel_train = pd.read_csv("./data/unlabeledTrainData.tsv", delimiter="\t", quoting=3)
        corpora = []
        for review in tqdm(unlabel_train['review']):
            corpora += self.review_to_sentences(review.decode('utf8'), tokenizer)
        return corpora

    def review_to_text(self, review, remove_stopwords=True):
        raw_text = BeautifulSoup(review, 'lxml').get_text()
        letters = re.sub('[^a-zA-Z]', ' ', raw_text)
        words = letters.lower().split()
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        return words

    def review_to_sentences(self, review):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.review_to_text(raw_sentence, False))
        return sentences

    def prepare_sequence(self, seq, to_ix, cuda=False):
        var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
        return var

    def prepare_label(self, label,label_to_ix, cuda=False):
        var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
        return var

    def build_token_to_ix(self, sentences):
        token_to_ix = dict()
        # print(len(sentences))
        for sent in sentences:
            for token in sent.split(' '):
                if token not in token_to_ix:
                    token_to_ix[token] = len(token_to_ix)
        token_to_ix['<pad>'] = len(token_to_ix)
        return token_to_ix

    def build_label_to_ix(self, labels):
        label_to_ix = dict()
        for label in labels:
            if label not in label_to_ix:
                label_to_ix[label] = len(label_to_ix)


class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.
    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tfile, root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = 'rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
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
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'rt-polarity.neg'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with codecs.open(os.path.join(path, 'rt-polarity.pos'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
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
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples

        if shuffle:
            random.shuffle(examples)

        dev_index = -1 * int(dev_ratio*len(examples))
        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


def load_mr(text_field, label_field, batch_size):
    train_data, dev_data = MR.splits(text_field, label_field)
    print (216, type(train_data), len(dev_data))
    # text_field.build_vocab(train_data, vectors="glove.6B.100d")
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    print ("building batches ...")
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        sort_key=lambda x: len(x.text),
        repeat=False, device=-1
    )
    return train_iter, dev_iter
#
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_iter, dev_iter , test_iter = load_mr(text_field, label_field, batch_size=50)


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)
