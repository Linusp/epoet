# coding: utf-8

import json
import numpy as np

from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.optimizers import RMSprop


def generate_from_corpus(corpus, batch_size):
    dict_size = len(corpus.dict_)
    EOS = np.zeros((batch_size, 1, dict_size))
    EOS[:] = np_utils.to_categorical([0], dict_size)

    for data in corpus.generator(batch_size):
        X = np.hstack((EOS, data))
        Y = np.hstack((data, EOS))

        yield (X, Y)


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
        self.model_.compile(loss="categorical_crossentropy", optimizer='adam')
        self.model_.load_weights(weights_file)
        self.dict_ = json.load(open(dict_file, 'r'))
        self.inv_dict_ = {v: k for k, v in self.dict_.iteritems()}

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
        self.model_.add(GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=True))
        self.model_.add(TimeDistributed(Dense(output_dim=input_size, activation="softmax")))
        self.model_.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.1))

    def train(self, corpus, batch_size=128, epoch=10):
        """corpus is a list of sentences."""
        self.dict_ = {k: v for k, v in corpus.dict_.iteritems()}
        self.inv_dict_ = {v: k for k, v in self.dict_.iteritems()}

        self.model_.fit_generator(generate_from_corpus(corpus, 128), 128, 100)

    def sample(self, begin_symbol=None, length=10):
        
