"""
    This script builds vocabulary dict
"""

# Notice
#   - use 'UPPER CASE' only
#   - use 'Character' only

import os 
import codecs

def load_all_files(fns):
    all_texts = []
    all_targets = []

    for fn in fns:
        fn = os.path.join( os.path.dirname(__file__), fn ) 

               
        with codecs.open(fn, 'r', encoding='utf-8') as f:
            data = [ sent.rstrip('\n').split('\t') for sent in f.readlines() ]


        all_targets += [ x[0] for x in data ]
        all_texts   += [ x[1] for x in data ]

    return all_targets, all_texts

def normalize(data):
    return [ sent.upper() for sent in data ]

from collections import Counter
def get_token_vocab_and_dump(data):
    counter = Counter()

    for sent in data:
        for char in sent:
            counter[char] += 1

    _dict = [
                '_UNK',
                '_PAD'
            ]

    for char, freq in counter.most_common():
        _dict.append( char ) 

    to_fn = os.path.join( os.path.dirname(__file__), 'token.vocab.txt' ) 
    with codecs.open(to_fn, 'w', encoding='utf-8') as f:
        for idx, c in enumerate(_dict):
            print("{}\t{}".format(c, idx), file=f)


def get_target_vocab_and_dump(data):
    counter = Counter()

    for t_class in data:
        counter[t_class] += 1

    _dict = []

    for char, freq in counter.most_common():
        _dict.append( char ) 

    to_fn = os.path.join( os.path.dirname(__file__), 'target.vocab.txt' ) 
    with codecs.open(to_fn, 'w', encoding='utf-8') as f:
        for idx, c in enumerate(_dict):
            print("{}\t{}".format(c, idx), file=f)


if __name__ == '__main__':
    fns = [ 'train.sent_data.txt', 'test.sent_data.txt']
    all_targets, all_texts = load_all_files(fns)
    
    target_data = normalize(all_targets)
    text_data   = normalize(all_texts)

    get_target_vocab_and_dump(target_data)
    get_token_vocab_and_dump(text_data)
