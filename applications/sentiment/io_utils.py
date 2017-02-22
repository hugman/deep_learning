#-*- coding: utf-8 -*-
"""
    Input / Output scripts collection

    Author : Sangkeun Jung (hugmanskj@gmail.com, 2017)
"""
# -- python 2/3 compatibility --- #
from __future__ import print_function
from __future__ import unicode_literals   # at top of module
from __future__ import absolute_import    # To make Py2 code safer (more like Py3) by preventing implicit relative imports, you can also add this to the top:
# ------------------------------- #

import os, sys, codecs
import random
import numpy as np 

class Vocabulary(object):
    """
        Load vocabulary from file
    """
    def __init__(self):
        self._token_to_id    = {}
        self._id_to_token    = {}

        self._class_to_id    = {}
        self._id_to_class    = {}

        self._num_tokens     = 0
        self._s_id           = None
        self._unk_id         = None
        self._pad_id         = None

        self._pos_id         = None
        self._neg_id         = None
        self._neu_id         = None
        self._obj_id         = None

    @property
    def s(self):
        return "<S>"

    @property
    def s_id(self):
        return self._s_id

    @property
    def unk(self):
        return "_UNK"

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def pad(self):
        return "_PAD"

    @property
    def pad_id(self):
        return self._pad_id

    # ----- polarity tags ----- #
    @property
    def POSITIVE_id(self):
        return self._pos_id

    @property
    def POSITIVE(self):
        return "POS"

    @property
    def NEGATIVE_id(self):
        return self._neg_id

    @property
    def NEGATIVE(self):
        return "NEG"

    @property
    def NEUTRAL_id(self):
        return self._neu_id

    @property
    def NEUTRAL(self):
        return "NEU"

    @property
    def OBJECTIVE_id(self):
        return self._obj_id

    @property
    def OBJECTIVE(self):
        return "OBJ"

    @property
    def num_tokens(self):
        self._num_tokens = len( self._token_to_id.keys() )
        return self._num_tokens 
    
    def get_id(self, token):
        return self._token_to_id.get(token, self.unk_id) # exist? return value, not exist? return unk

    def get_class_id(self, sym):
        return self._class_to_id.get(sym) 

    def get_token(self, id):
        return self._id_to_token[id]

    def get_class_symbol(self, id):
        return self._id_to_class[id]

    def finalize(self):
        self._unk_id = self.get_id(self.unk)
        self._pad_id = self.get_id(self.pad)

        self._pos_id = self.get_class_id(self.POSITIVE)
        self._neg_id = self.get_class_id(self.NEGATIVE)
        self._neu_id = self.get_class_id(self.NEUTRAL)
        self._obj_id = self.get_class_id(self.OBJECTIVE)
        

    def add(self, token, id):
        self._token_to_id[token] = id
        self._id_to_token[id]    = token

    def add_class(self, sym, id):
        self._class_to_id[sym] = id
        self._id_to_class[id]  = sym

    @staticmethod
    def from_file(filename):
        vocab = Vocabulary()
        with codecs.open(filename, "r", "utf-8") as f:
            for _idx, line in enumerate(f):
                word, count = line.rstrip('\n').split('\t')
                vocab.add(word, _idx)
        _target = ['POS', 'NEG', 'NEU', 'OBJ'] 
        for _idx, sym in enumerate( _target ):
            vocab.add_class(sym, _idx) 
        vocab.finalize()
        return vocab


class Dataset(object):
    def __init__(self, data_fn, vocab, deterministic=False):
        # if deterministic=True? data will be shuffled
        self._data_fn       = data_fn
        self._vocab         = vocab
        self._deterministic = deterministic
        self.data_files     = [ self._data_fn ]

    def _parse_sentence(self, line):
        s_id = self._vocab.s_id
        polarity, sentence = line.rstrip('\n').split('\t')

        #sent_ids  = [s_id] 
        sent_ids = [self._vocab.get_id(token) for token in sentence]  # characterwise
        #sent_ids += [s_id] 

        pol_id = self._vocab.get_class_id(polarity)
        return pol_id, sent_ids

    def _parse_file(self, file_name):
        print("Processing file: %s" % file_name)
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.rstrip('\n') for line in f]
            if not self._deterministic:
                random.shuffle(lines)
            print("Finished processing!")

            for line in lines:
                yield self._parse_sentence(line)

    def _sentence_stream(self, file_stream):
        for file_name in file_stream:
            for sent_info in self._parse_file(file_name):
                yield sent_info

    def _iterate(self, sent_info, batch_size, max_len):
        # vectorize id data
        streams = [None] * batch_size

        x = np.zeros([batch_size, max_len], np.int32)  # Integer (-2147483648 to 2147483647) - for sentence
        y = np.zeros([batch_size],          np.uint8)  # Unsigned integer (0 to 255)         - for polarity
        w = np.zeros([batch_size, max_len], np.uint8)  # Unsigned integer (0 to 255)         - for pad symbol recording
        while True:
            x[:] = 0
            y[:] = 0
            w[:] = 0
            for b in range(batch_size):
                try:
                    polarity_id, sent_ids = next(sent_info)

                    pad_ids = [ self._vocab.pad_id ] * (max_len - len(sent_ids))
                    pad_markers = [ 0 ] * (max_len - len(sent_ids))

                    x[b]    = sent_ids + pad_ids
                    y[b]    = polarity_id
                    w[b]    = [ 1 ] * len(sent_ids) + pad_markers
                    
                except StopIteration:
                    pass
            if not np.any(w):
                return

            # x : sentence 
            # y : poloarity
            # w : pad markers
            yield x, y, w

    def iterate_once(self, batch_size, num_steps):
        def file_stream():
            for file_name in self.data_files:
                yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value
    
    def iterate_forever(self, batch_size, num_steps):
        def file_stream():
            while True:
                if not self._deterministic:
                    random.shuffle(self.data_files)
                for file_name in self.data_files:
                    yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value # value : tuple ( polairty_id, sent_ids )


    def recover_sentence(self, x):
        sentence = [ self._vocab.get_token(id) for id in x ] 
        return sentence 

    def recover_class(self, y):
        # ids to symbols
        _class = self._vocab.get_class_symbol(y)
        return _class
