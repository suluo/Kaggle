############################################
# File Name    : train_batch.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2018-03-08
# Last Modified: 2018-04-03 19:24:29
# Descption    :
# Version      : Python 3.6
############################################
from __future__ import division
import argparse
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.autograd import Variable
import random
import time
import logging
import logging.config
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

logging.config.fileConfig('./conf/logging.conf',
                          disable_existing_loggers=False)
logger = logging.getLogger(__file__)


# load word embedding
def load_my_vecs(path, vocab, freqs):
    word_vecs = {}
    with open(path, encoding="utf-8") as f:
        count = 0
        lines = f.readlines()[1:]
        for line in lines:
            values = line.split(" ")
            word = values[0]
            # word = word.lower()
            count += 1
            if word in vocab:  # whether to judge if in vocab
                vector = []
                for count, val in enumerate(values):
                    if count == 0:
                        continue
                    vector.append(float(val))
                word_vecs[word] = vector
    return word_vecs


def train(model, train_iter, dev_iter, best_dev_acc=0.0):
    if torch.cuda.is_available():
        model = model.cuda()

    for i in range(10):
        t0 = time.time()
        logger.info('Epoch : %s start ...' % i)
        train_epoch(model, train_iter, epoch=i)
        dev_acc = train_epoch(model, dev_iter, False)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print('New Best Dev - {} !!!'.format(best_dev_acc))
            torch.save(model.state_dict(), "./data/net_params.pkl")
        logger.info('epoch %s taks %s h, now best dev acc %s'
                    % (i, (time.time()-t0)/3600, best_dev_acc))


def train_epoch(model, train_iter, is_train=True, epoch=0):
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2)

    if is_train:
        model.train()
    else:
        model.eval()
    avg_loss, corrects, count = 0.0, 0.0, 0

    for batch in tqdm(train_iter):
        sents, labels = batch.text, batch.label.data.sub_(1)
        # feature, target = Variable(sents), Variable(labels)
        feature, target = sents, Variable(labels)
        if torch.cuda.is_available():
            feature, target = feature.cuda(), target.cuda()

        model.batch_size = batch.batch_size
        model.hidden = model.init_hidden()

        logit = model(feature)
        loss = loss_func(logit, target)

        batch_correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        avg_loss += loss.data[0]
        if is_train:
            # should I keep this when I am evaluating the model?
            # model.zero_grad()
            # optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 200 == 0:
                # print('train-epoch: {} iterations: {} loss: {}'.format(epoch, count*model.batch_size, loss.data[0]))
                logger.info('train-epoch: %s iterations: %s loss: %s acc: %s'
                            % (epoch, count*model.batch_size, loss.data[0], float(batch_correct)/model.batch_size))

    train_size = len(train_iter.dataset)
    avg_loss /= train_size
    accuracy = float(corrects)/train_size
    logger.info('Is_train: %s epoch: %s avg_loss: %s, acc: %s'
                % (is_train, epoch, avg_loss, accuracy))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def main(num):
    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args.num)
