import numpy as np
import os, sys

from common.nlp.vocab import Vocab
from common.nlp.dataset import Dataset
from common.nlp.converter import N2MConverter
from common.nlp.data_loader import N2MTextData
import copy


# for Chitchat dataset
class ChitchatDataset(Dataset):
    def _iterate(self, index_gen):
        B = self.batch_size

        SN = self.src_num_steps
        TN = self.tar_num_steps

        # vectorize id data
        src_token  = np.zeros([B, SN], np.int64)    # x
        tar_token  = np.zeros([B, TN], np.int64)    # y 
        
        src_weight = np.zeros([B, SN], np.float32)  # sw
        tar_weight = np.zeros([B, TN], np.float32)  # tw

        while True:
            src_token[:]      = 0
            tar_token[:]      = 0

            src_weight[:]     = 0
            tar_weight[:]     = 0

            for b in range(B):
                try:
                    while True:
                        index = next(index_gen)
                        src_num_steps = len( self.data[index].src_token_ids )
                        tar_num_steps = len( self.data[index].tar_token_ids )

                        if src_num_steps <= SN and tar_num_steps <= TN: break 

                    src_token_ids    = copy.deepcopy( self.data[index].src_token_ids )
                    tar_token_ids    = copy.deepcopy( self.data[index].tar_token_ids )

                    # fill pad for src_weight
                    src_weight_ids = [0] * self.src_num_steps
                    for _idx, _ in enumerate(src_token_ids): src_weight_ids[_idx] = 1

                    # fill pad for tar_weight
                    tar_weight_ids = [0] * self.tar_num_steps
                    for _idx, _ in enumerate(tar_token_ids): tar_weight_ids[_idx] = 1


                    # input
                    # fill pad to src token
                    src_token_ids += [self.token_pad_id] * ( self.src_num_steps - len( src_token_ids ) ) 

                    # output
                    # fill pad to tar token
                    tar_token_ids += [self.token_pad_id] * ( self.tar_num_steps - len( tar_token_ids ) ) 

                    # input
                    src_token[b] = src_token_ids
                    tar_token[b] = tar_token_ids

                    src_weight[b] = src_weight_ids
                    tar_weight[b] = tar_weight_ids

                except StopIteration:
                    pass
            if not ( np.any(src_weight) or np.any(tar_weight) ):
                return
            yield src_token, tar_token, src_weight, tar_weight

    def unit_test(self):
        a_batch_data = next(self.iterator)
        x, y, sw, tw = a_batch_data
        print(x, y, sw, tw)


def load_data():

    # vocab loader
    src_token_vocab_fn  = os.path.join( os.path.dirname(__file__), 'data', 'token.source.vocab.txt')
    src_token_vocab     = Vocab(src_token_vocab_fn, mode='token')
    tar_token_vocab_fn  = os.path.join( os.path.dirname(__file__), 'data', 'token.target.vocab.txt')
    tar_token_vocab     = Vocab(tar_token_vocab_fn, mode='token')

    # load train data 
    train_data_fn  = os.path.join( os.path.dirname(__file__), 'data', 'chitchat.n2m.txt')
    train_txt_data = N2MTextData(train_data_fn)

    # convert text data to id data
    train_id_data  = N2MConverter.convert(train_txt_data, src_token_vocab, tar_token_vocab)
    
    return train_id_data, src_token_vocab, tar_token_vocab


