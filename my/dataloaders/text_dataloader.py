import os
import sys
import numpy as np
import random

import tensorflow as tf
from dataloaders.text_featurizers import TextFeaturizer

class Text_Dataloader():
    def __init__(self, config, training=True):
        #self.text_featurizer = TextFeaturizer(config)
        self.batch = config['batch']
        self.make_dict(config['lexicon_p'])
        self.make_egs_list(config['train_p'] if training else config['eval_p'], config['ngram'], training)
        self.epochs = 1

    def make_dict(self, lexicon_p):
        self.word2index = {}
        self.index2word = {}
        with open(lexicon_p, 'r', encoding='utf-8') as f:
            for line in f:
                index, word = line.strip().split(' ')
                self.word2index[word] = index
                self.index2word[index] = word

    #输入的文本是已经分词的
    def make_egs_list(self, text_p, ngram, training):
        data = []
        with open(text_p, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split(' ')
                words = [self.word2index[word] for word in words]
                for i in range(len(words) - ngram):
                    data.append(words[i:i+ngram])
        num = len(data)
        if training:
            self.train_list = data[:int(num * 0.99)]
            self.test_list = data[int(num * 0.99):]
            np.random.shuffle(self.train_list)
            self.pick_index = [0.] * len(self.train_list)
        else:
            self.test_list = data
            
    def generate(self, train=True):
        if train:
            batch = self.batch
            indexs = np.argsort(self.pick_index)[:batch] 
            indexs = random.sample(indexs.tolist(), batch)
            sample = [self.train_list[i] for i in indexs]
            for i in indexs:
                self.pick_index[int(i)] += 1
            self.epochs = 1 + int(np.mean(self.pick_index))
        else:
            sample = random.sample(self.test_list, self.batch)

        x_l = []
        y_l = []
        for words in sample:
            x = words[:-1]
            y = words[-1]
            x_l.append(np.array(x))
            y_l.append(np.array(y))
        return np.array(x_l), np.array(y_l)

    def generator(self,training=True):
        while 1:
            x_l, y_l = self.generate(training)
            yield x_l, y_l

    def return_data_types(self):
        return (tf.int32, tf.int32)

    def return_data_shape(self):
        return (tf.TensorShape([None, None]), tf.TensorShape([None,]))
