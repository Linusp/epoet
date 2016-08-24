# coding: utf-8

import re
import pandas as pd
import numpy as np

from epoet.corpus import Corpus, PairCorpus
from epoet.rnnlm import generate_from_corpus
from epoet.poet import build_model, save_model_to_file, train_model
from seq2seq.models import Seq2seq, AttentionSeq2seq


origin_data = pd.read_csv('data/tang_poetries.csv').content.tolist()
corpus = []
pair_corpus = []

for poetry in origin_data:
    poetry = poetry.decode('utf-8').strip(u'，。')
    last_sent = u''
    for sent in poetry.split(u'。'):
        sent = sent.strip()
        if sent == u'':
            continue

        comps = filter(lambda x: x.strip() != u'', sent.split(u'，'))
        corpus.extend(comps)

        for i in range(len(comps) - 1):
            pair_corpus.append((comps[i], comps[i+1]))

        if last_sent != u'':
            pair_corpus.append((last_sent, u''.join(comps)))

        last_sent = u''.join(comps)

corpus_obj = Corpus(corpus)
pair_corpus_obj = PairCorpus(pair_corpus)

print corpus_obj.max_len
print pair_corpus_obj.max_len


j = 0
for batch_x, batch_y in generate_from_corpus(corpus_obj, 10, corpus_obj.max_len - 1):
    for i in range(batch_x.shape[0]):
        # print np.argmax(batch_x[i], axis=1)
        # print np.argmax(batch_y[i])
        j += 1
        if j >= 3:
            break

    if j >= 3:
        break


# for batch in corpus_obj.generator(1):
#     print batch.shape
#     break
