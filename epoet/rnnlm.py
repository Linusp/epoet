# coding: utf-8

import json
import numpy as np

from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint


def generate_from_corpus(corpus, batch_size, seq_len):
    dict_size = len(corpus.dict_)
    EOS_ONEHOT = np_utils.to_categorical([corpus.inv_dict_[corpus._EOS_]], dict_size)

    X, y = [], []

    j = 0
    for data in corpus.generator(1):
        d = data[0]
        for i in range(len(d) - 1):
            cur_x = d[:i+1]
            cur_y = d[i+1]

            if (EOS_ONEHOT - cur_y).any():
                X.append(cur_x)
                y.append(cur_y)
                j += 1

            if j == batch_size:
                X = pad_sequences(X, seq_len, value=EOS_ONEHOT)
                y = np.array(y).reshape(batch_size, dict_size)

                yield (X, y)

                X, y = [], []
                j = 0


class LanguageModel(object):

    _MODEL_STRUCT_ = '_struct.json'
    _MODEL_WEIGHTS_ = '_weights.h5'
    _MODEL_DICT_ = '_dict.json'
    _EOS_ = u'^'

    def __init__(self):
        self.model_ = None
        self.dict_ = {}
        self.inv_dict_ = {}

    def load_model(self, model_prefix):
        struct_file = model_prefix + self._MODEL_STRUCT_
        weights_file = model_prefix + self._MODEL_WEIGHTS_
        dict_file = model_prefix + self._MODEL_DICT_

        self.model_ = model_from_json(open(struct_file, 'r').read())
        self.model_.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
        self.model_.load_weights(weights_file)
        self.dict_ = {int(k): v for k, v in json.load(open(dict_file, 'r')).iteritems()}
        self.inv_dict_ = {v: int(k) for k, v in self.dict_.iteritems()}

    def save_model(self, model_prefix):
        struct_file = model_prefix + self._MODEL_STRUCT_
        weights_file = model_prefix + self._MODEL_WEIGHTS_
        dict_file = model_prefix + self._MODEL_DICT_

        model_struct = self.model_.to_json()
        open(struct_file, 'w').write(model_struct)
        self.model_.save_weights(weights_file, overwrite=True)
        json.dump(self.dict_, open(dict_file, 'w'))

    def build_model(self, input_size, hidden_size):
        self.model_ = Sequential()
        self.model_.add(GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
        self.model_.add(Dense(output_dim=input_size, activation='softmax'))
        self.model_.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def train(self, corpus, batch_size=128, nb_epoch=10, model_prefix="./lm"):
        """corpus is a list of sentences."""
        self.dict_ = {k: v for k, v in corpus.dict_.iteritems()}
        self.inv_dict_ = {v: k for k, v in self.dict_.iteritems()}

        checkpointer = ModelCheckpoint(filepath=model_prefix + "_tmp" + self._MODEL_WEIGHTS_, monitor="accuracy")
        self.model_.fit_generator(
            generate_from_corpus(corpus, batch_size, corpus.max_len - 1),
            batch_size,
            nb_epoch,
            callbacks=[checkpointer]
        )

        self.save_model(model_prefix)

    def sample(self, begin_symbol=None, length=10):
        x = [self.inv_dict_[self._EOS_]]
        if begin_symbol:
            x.append(self.inv_dict_[begin_symbol])

        while len(x) <= length:
            cur_x = np_utils.to_categorical(x, len(self.dict_))
            cur_x = cur_x.reshape((1, cur_x.shape[0], cur_x.shape[1]))
            x.append(np.argmax(self.model_.predict(cur_x)))

        return u''.join([self.dict_[idx] for idx in x if idx != 0])
