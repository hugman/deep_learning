
import os, sys, codecs
import copy

class N21Item:
    def __init__(self, target, text):
        self.target = target
        self.text   = text

        self.target_id   = None
        self.token_ids   = None  # it should be array 

    def set_id(self, target_id, token_ids):
        self.target_id = target_id
        self.token_ids = token_ids

    def get_tokens(self):
        # currently, only support 'character' 
        return list(self.text) 


class N2NItem:
    def __init__(self, targets, tokens):
        self.text   = "".join(tokens)
        self.target_txts = targets
        self.token_txts  = tokens

        self.target_ids  = None
        self.token_ids   = None  # it should be array 

    def set_id(self, target_ids, token_ids):
        self.target_ids = target_ids
        self.token_ids  = token_ids

    def get_tokens(self):
        # currently, only support 'character' 
        return list(self.text) 



class N21TextData:
    def __init__(self, src=None, mode='file'):  # mode = 'file' | 'sentence'
        self.data = []
        
        if mode == 'file':      self.load_text_file_data(src)
        if mode == 'sentence':  self.load_text_data(src)

    def add_to_data(self, target, text):
        # normalize
        target = target.upper()
        text   = text.upper()
        self.data.append( N21Item(target, text) )

    def load_text_data(self, line):
        # mode = 'sentence'
        # format of line : "TAG  \t  SENTENCE"
        line = line.rstrip('\n\r')
        target, text = line.split('\t')
        self.add_to_data(target, text)

    def load_text_file_data(self, fn):
        # mode = 'file' 
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')
                target, text = line.split('\t')

                self.add_to_data(target, text)



class N2NTextData:
    def __init__(self, src=None, mode='file'):  # mode = 'file' | 'sentence'
        self.data = []
        
        if mode == 'file':      self.load_text_file_data(src)
        if mode == 'sentence':  self.load_text_data(src)

    def add_to_data(self, targets, tokens):
        # normalize
        targets = [ t.upper() for t in targets ]
        tokens  = [ t.upper() for t in tokens ] 
        self.data.append( N2NItem(targets, tokens) )

    def load_text_data(self, line):
        # mode = 'sentence'
        line = line.rstrip('\n\r')
        tokens  = list(line)
        targets = [ 'O' for x in tokens ]
        self.add_to_data(targets, tokens)


    def load_text_file_data(self, fn):
        # mode = 'file' 
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            a_sent_data = []
            tokens = []
            targets = []
            for line in f:
                line = line.rstrip('\n\r')

                if line.startswith('-------'): 
                    self.add_to_data(targets, tokens)
                    tokens = []
                    targets = []
                    continue 

                fields = line.split('\t')
                assert (len(fields) >=  2), "Not implemented spec"
                assert (len(fields) == 2), "Wrong data format "

                token, tag = fields[0], fields[1]
                tokens.append( token ) 
                targets.append( tag ) 


    


    

