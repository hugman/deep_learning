
import os, sys, codecs

import copy 
class N21Converter:

    @staticmethod
    def convert(txt_data, target_vocab, token_vocab):
        # txt_data : it should be N21TextData
        # target_vocab    : it should be Vocab
        # token_vocab    : it should be Vocab

        id_data = []  # it should be list of N21Item

        for item in txt_data.data:
            target_id = target_vocab.get_id(item.target)
            text_tokens = item.get_tokens()

            token_ids = [ token_vocab.get_id(token) for token in text_tokens ] # for each token

            new_item = copy.deepcopy(item)
            new_item.set_id(target_id,token_ids)

            id_data.append( new_item )
        return id_data


class N2NConverter:

    @staticmethod
    def convert(txt_data, target_vocab, token_vocab):
        # txt_data : it should be N21TextData
        # target_vocab    : it should be Vocab
        # token_vocab    : it should be Vocab

        id_data = []  # it should be list of N21Item

        for item in txt_data.data:

            # targets
            target_ids = []
            for target in item.target_txts:
                target_id = target_vocab.get_id(target)
                target_ids.append( target_id ) 

            # tokens
            token_ids = []
            for token in item.token_txts:
                token_id = token_vocab.get_id(token)
                token_ids.append( token_id ) 

            new_item = copy.deepcopy(item)
            new_item.set_id(target_ids, token_ids)
            id_data.append( new_item )

        return id_data





    

