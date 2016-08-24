# coding: utf-8

import re
import pandas as pd

from epoet.corpus import Corpus, PairCorpus
from epoet.rnnlm import generate_from_corpus, LanguageModel
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

print len(corpus), corpus_obj.max_len, len(corpus_obj.dict_)
# print pair_corpus_obj.max_len

model = LanguageModel()
# model.load_model('models/lm')
# model.sample(u'白')
model.build_model(len(corpus_obj.dict_), 400)
model.train(corpus_obj, 512, 1000, model_prefix="models/lm")

# # model = build_model(len(pair_corpus_obj.dict_), pair_corpus_obj.max_len, 1024)
# # train_model(model, pair_corpus_obj)

# model = Seq2seq(
#     batch_input_shape=(128, pair_corpus_obj.max_len, len(pair_corpus_obj.dict_)),
#     # input_dim=len(pair_corpus_obj.dict_),
#     # input_length=pair_corpus_obj.max_len,
#     hidden_dim=256,
#     output_length=pair_corpus_obj.max_len,
#     output_dim=len(pair_corpus_obj.dict_)
# )
# model.compile(loss='mse', optimizer='rmsprop')

# model.fit_generator(pair_corpus_obj.generator(128), 512, 2000)
# save_model_to_file(model, 'seq2seq_struct.json', 'seq2seq_weights.h5')
