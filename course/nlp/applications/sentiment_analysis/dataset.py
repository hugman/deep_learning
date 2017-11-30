import numpy as np
import os, sys

from common.nlp.vocab import Vocab
from common.nlp.dataset import Dataset
from common.nlp.converter import N21Converter
from common.nlp.data_loader import N21TextData
import copy


# for sentiment dataset
class SentimentDataset(Dataset):
    def _iterate(self, index_gen):
        B = self.batch_size
        N = self.num_steps

        # vectorize id data
        sentiment  = np.zeros([B],    np.int64)  
        token      = np.zeros([B, N], np.int64)
        weight     = np.zeros([B, N], np.int64)

        while True:
            sentiment[:]  = 0
            token[:]      = 0
            weight[:]     = 0

            for b in range(B):
                try:
                    while True:
                        index = next(index_gen)
                        _num_steps = len( self.data[index].token_ids )
                        if _num_steps <= N: break 

                    _sentiment_id = copy.deepcopy( self.data[index].target_id )
                    _token_ids    = copy.deepcopy( self.data[index].token_ids )

                    # fill pad for weight
                    _weight_ids   = [0] * self.num_steps
                    for _idx, _ in enumerate(_token_ids): _weight_ids[_idx] = 1

                    # fill pad to token
                    _token_ids += [self.token_pad_id] * ( self.num_steps - len( _token_ids ) ) 

                    # output
                    sentiment[b] = -1 if _sentiment_id is None else _sentiment_id

                    # input
                    token[b]  = _token_ids
                    weight[b] = _weight_ids

                except StopIteration:
                    pass
            if not np.any(weight):
                return
            yield sentiment, token, weight  # tuple for (target, input)


def load_data():

    # vocab loader
    token_vocab_fn  = os.path.join( os.path.dirname(__file__), 'data', 'token.vocab.txt')
    token_vocab     = Vocab(token_vocab_fn, mode='token')
    target_vocab_fn = os.path.join( os.path.dirname(__file__), 'data', 'target.vocab.txt')
    target_vocab    = Vocab(target_vocab_fn, mode='target')

    # load train data 
    #train_data_fn  = os.path.join( os.path.dirname(__file__), 'data', 'train.sent_data.txt')
    train_data_fn  = os.path.join( os.path.dirname(__file__), 'data', 'small.train.sent_data.txt')
    train_txt_data = N21TextData(train_data_fn)

    # convert text data to id data
    train_id_data  = N21Converter.convert(train_txt_data, target_vocab, token_vocab)
    
    return train_id_data, token_vocab, target_vocab