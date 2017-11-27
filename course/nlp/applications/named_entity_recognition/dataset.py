import numpy as np
import os, sys

from common.nlp.vocab import Vocab
from common.nlp.dataset import Dataset
from common.nlp.converter import N2NConverter
from common.nlp.data_loader import N2NTextData
import copy


# for NER dataset
class NERDataset(Dataset):
    def _iterate(self, index_gen):
        B = self.batch_size
        N = self.num_steps

        # vectorize id data
        ne         = np.zeros([B, N], np.int64)    # y 
        token      = np.zeros([B, N], np.int64)    # x
        weight     = np.zeros([B, N], np.float32)  # w

        while True:
            ne[:]         = 0
            token[:]      = 0
            weight[:]     = 0

            for b in range(B):
                try:
                    while True:
                        index = next(index_gen)
                        _num_steps = len( self.data[index].token_ids )
                        if _num_steps <= N: break 

                    _ne_ids       = copy.deepcopy( self.data[index].target_ids )

                    _token_ids    = copy.deepcopy( self.data[index].token_ids )

                    # fill pad for weight
                    _weight_ids   = [0] * self.num_steps
                    for _idx, _ in enumerate(_token_ids): _weight_ids[_idx] = 1

                    # fill O id to target
                    _ne_ids += [self.target_null_id] * ( self.num_steps - len( _token_ids ) ) 

                    # fill pad to token
                    _token_ids += [self.token_pad_id] * ( self.num_steps - len( _token_ids ) ) 

                    # output
                    ne[b] = _ne_ids

                    # input
                    token[b]  = _token_ids
                    weight[b] = _weight_ids
                    ne[b]     = _ne_ids

                except StopIteration:
                    pass
            if not np.any(weight):
                return
            yield ne, token, weight  # tuple for (target, input)

    def unit_test(self):
        a_batch_data = next(self.iterator)
        y, x, w = a_batch_data
        print(y,x,w)




def load_data():

    # vocab loader
    token_vocab_fn  = os.path.join( os.path.dirname(__file__), 'data', 'token.vocab.txt')
    token_vocab     = Vocab(token_vocab_fn, mode='token')
    target_vocab_fn = os.path.join( os.path.dirname(__file__), 'data', 'target.vocab.txt')
    target_vocab    = Vocab(target_vocab_fn, mode='target')

    # load train data 
    train_data_fn  = os.path.join( os.path.dirname(__file__), 'data', 'ner.n2n.txt')
    train_txt_data = N2NTextData(train_data_fn)

    # convert text data to id data
    train_id_data  = N2NConverter.convert(train_txt_data, target_vocab, token_vocab)
    
    return train_id_data, token_vocab, target_vocab