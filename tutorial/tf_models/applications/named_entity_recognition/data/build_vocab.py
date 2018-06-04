"""
    This script builds vocabulary dict
"""

# Notice
#   - use 'UPPER CASE' only
#   - use 'Character' only

import os 
import codecs
import collections

def normalize(data):
    return [ sent.upper() for sent in data ]


def load_all_dump_vocab(fns):
    # character level processing
    all_tokens  = collections.Counter()
    all_targets = collections.Counter()

    for fn in fns:
        fn = os.path.join( os.path.dirname(__file__), fn ) 
        with codecs.open(fn, 'r', encoding='utf-8') as f:

            for line in f:
                if line.startswith('---------'): 
                    # a sentence end
                    continue 

                line = line.rstrip('\n\r')
                token, target = line.split('\t')

                all_tokens[ token.upper() ] += 1 # character
                all_targets[ target ] += 1 

    # dump tokens
    _dict = [
                '_UNK',
                '_PAD'
            ]
    for char, freq in all_tokens.most_common():
        _dict.append( char ) 

    to_fn = os.path.join( os.path.dirname(__file__), 'token.vocab.txt' ) 
    with codecs.open(to_fn, 'w', encoding='utf-8') as f:
        for idx, c in enumerate(_dict):
            print("{}\t{}".format(c, idx), file=f)
    print("Token vocab file is dumped at {}".format(to_fn) )

    # dump targets
    _dict = []

    for tag, freq in all_targets.most_common():
        _dict.append( tag ) 

    to_fn = os.path.join( os.path.dirname(__file__), 'target.vocab.txt' ) 
    with codecs.open(to_fn, 'w', encoding='utf-8') as f:
        for idx, tag in enumerate(_dict):
            print("{}\t{}".format(tag, idx), file=f)
    print("Target vocab file is dumped at {}".format(to_fn) )


if __name__ == '__main__':
    fns = [ 'ner.n2n.txt' ]
    load_all_dump_vocab(fns)
