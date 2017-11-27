
import os, sys, codecs


class Vocab:
    def __init__(self, fn, mode='token'):
        self.mode = mode 
        self.token_unk_id = 0
        self.token_unk_symbol = '_UNK'

        self.token_pad_id = 1
        self.token_pad_symbol = '_PAD'
        
        self.token_2_id = {}
        self.id_2_token = {}

        self.target_2_id = {}
        self.id_2_target = {}

        self.target_out_symbol = 'O'
        
        if mode == 'token'  : self.load_token_vocab(fn)
        if mode == 'target' : self.load_target_vocab(fn)

    def load_token_vocab(self, fn):
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')
                token, id = line.split('\t')

                id = int(id)

                if token == self.token_unk_symbol: 
                    self.token_2_id[token] = self.token_unk_id
                    continue 

                if token == self.token_pad_symbol: 
                    self.token_2_id[token] = self.token_pad_id
                    continue 

                # other tokens
                self.token_2_id[token] = id
                self.id_2_token[id] = token 

    def load_target_vocab(self, fn):
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')
                target, id = line.split('\t')
                id = int(id)

                self.target_2_id[target] = id
                self.id_2_target[id] = target

    def get_id(self, symbol):
        if self.mode == 'token':
            return self.token_2_id.get(symbol, self.token_unk_id)

        if self.mode == 'target':
            return self.target_2_id.get(symbol)

    def get_symbol(self, id):
        if self.mode == 'token':
            return self.id_2_token.get(id)

        if self.mode == 'target':
            return self.id_2_target.get(id)

    def get_num_tokens(self):
        if self.mode == 'token': return len(self.token_2_id)
        return None 

    def get_num_targets(self):
        if self.mode == 'target': return len(self.target_2_id)
        return None 

    def get_token_pad_id(self):   return self.token_pad_id
    def get_target_null_id(self): return self.get_id(self.target_out_symbol)

    









    

