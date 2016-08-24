# coding: utf-8

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


class Corpus(object):

    _EOS_ = '^'

    def __init__(self, corpus_text):
        """corpus_text: list of text"""
        self.corpus_ = []
        self.dict_ = {0: self._EOS_}
        self.inv_dict_ = {self._EOS_: 0}

        self.build_from_corpus(corpus_text)
        self.max_len = max(len(sent) for sent in corpus_text) + 1

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
        EOS_ONEHOT = np_utils.to_categorical([self.inv_dict_[self._EOS_]], dict_size)
        data = []

        for i, sent in enumerate(self.corpus_):
            data.append(np_utils.to_categorical(sent, dict_size))

            if (i + 1) % batch_size == 0:
                yield pad_sequences(data, self.max_len, value=EOS_ONEHOT)
                data = []


class PairCorpus(object):
    _EOS_ = '^'

    def __init__(self, pair_corpus):
        """[(xxx, xxx), (xxx, xxx)]"""
        self.corpus_ = []
        self.dict_ = {0: self._EOS_}
        self.inv_dict_ = {self._EOS_: 0}

        self.build_from_corpus(pair_corpus)
        self.max_len = max(len(pair[0]) for pair in pair_corpus) + 1

    def build_from_corpus(self, pair_corpus):
        def vectorize_update_dict(sent):
            sent_vec = []
            for ch in sent:
                if ch not in self.inv_dict_:
                    cur_index = len(self.inv_dict_)
                    self.inv_dict_[ch] = cur_index
                    self.dict_[cur_index] = ch

                sent_vec.append(self.inv_dict_[ch])

            return sent_vec

        for sent_pair in pair_corpus:
            first = vectorize_update_dict(sent_pair[0])
            second = vectorize_update_dict(sent_pair[1])

            self.corpus_.append((first, second))

    def generator(self, batch_size=128):
        dict_size = len(self.dict_)
        EOS_ONEHOT = np_utils.to_categorical([self.inv_dict_[self._EOS_]], dict_size)

        data_x = []
        data_y = []

        i = 0
        corpus_length = len(self.corpus_)
        while True:
            sent_x, sent_y = self.corpus_[i % corpus_length]
            data_x.append(np_utils.to_categorical(sent_x, dict_size))
            data_y.append(np_utils.to_categorical(sent_y, dict_size))

            if (i + 1) % batch_size == 0:
                yield (
                    pad_sequences(data_x, self.max_len, value=EOS_ONEHOT),
                    pad_sequences(data_y, self.max_len, value=EOS_ONEHOT),
                )
                data_x = []
                data_y = []

            i += 1
