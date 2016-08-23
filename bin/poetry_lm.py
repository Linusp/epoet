# coding: utf-8

import re
import pandas as pd

from epoet.rnnlm import LanguageModel
from epoet.corpus import Corpus


origin_data = pd.read_csv('data/tang_poetries.csv').content.tolist()
corpus = []

for poetry in origin_data:
    poetry = poetry.decode('utf-8').strip(u'，。')
    for sent in re.split(u'[，。]', poetry):
        if len(sent) not in (5, 7):
            print poetry
        if sent.strip() != u'':
            corpus.append(sent)

data = Corpus(corpus)
lm = LanguageModel()
lm.build_model(len(data.dict_), 400)
lm.train(data)
