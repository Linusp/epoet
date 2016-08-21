# coding: utf-8

import numpy as np
from keras.utils import np_utils


class Corpus(object):

    _EOS_ = '^'

    def __init__(self, corpus_text):
        """corpus_text: list of text"""
        self.corpus_ = []
        self.dict_ = {0: self._EOS_}
        self.inv_dict_ = {self._EOS_: 0}

        self.build_from_corpus(corpus_text)
        self.max_len = max(len(sent) for sent in corpus_text)

    def build_from_corpus(self, corpus):
        for sent in corpus:
            sent_vec = []
            for ch in sent:
                if ch not in self.inv_dict_:
                    cur_index = len(self.inv_dict_)
                    self.inv_dict_[ch] = cur_index
                    self.dict_[cur_index] = ch

                sent_vec.append(self.inv_dict_[ch])

            self.corpus_.append(sent_vec)

    def generator(self, batch_size=128):
        dict_size = len(self.dict_)
        data = np.zeros((batch_size, self.max_len, dict_size))

        for i, sent in enumerate(self.corpus_):
            data[i % batch_size, :len(sent)] = np_utils.to_categorical(sent, dict_size)

            if len(sent) < self.max_len:
                remain_len = self.max_len - len(sent)
                data[i % batch_size, len(sent):] = np_utils.to_categorical([0] * remain_len, dict_size)

            if (i + 1) % batch_size == 0:
                yield data
