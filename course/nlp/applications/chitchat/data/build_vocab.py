"""
    This script builds vocabulary dict for n2m 
        - source text vocab
        - target text vocab 

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
    all_src_tokens = collections.Counter()
    all_tar_tokens = collections.Counter()

    for fn in fns:
        fn = os.path.join( os.path.dirname(__file__), fn ) 
        with codecs.open(fn, 'r', encoding='utf-8') as f:

            for line in f:
                line = line.rstrip('\n\r')
                src_text, tar_text = line.split('\t')

                for token in src_text:
                    all_src_tokens[ token.upper() ] += 1 # character
                for token in tar_text:
                    all_tar_tokens[ token.upper() ] += 1 # character


    # dump tokens
    src_dict = [
                '_UNK',
                '_PAD'
               ]

    tar_dict = [
                '_UNK',
                '_PAD'
               ]


    for char, freq in all_src_tokens.most_common():
        src_dict.append( char ) 

    for char, freq in all_tar_tokens.most_common():
        tar_dict.append( char ) 

    to_src_fn = os.path.join( os.path.dirname(__file__), 'token.source.vocab.txt' ) 
    with codecs.open(to_src_fn, 'w', encoding='utf-8') as f:
        for idx, c in enumerate(src_dict):
            print("{}\t{}".format(c, idx), file=f)
    print("Source text Token vocab file is dumped at {}".format(to_src_fn) )

    to_tar_fn = os.path.join( os.path.dirname(__file__), 'token.target.vocab.txt' ) 
    with codecs.open(to_tar_fn, 'w', encoding='utf-8') as f:
        for idx, c in enumerate(tar_dict):
            print("{}\t{}".format(c, idx), file=f)
    print("Target text Token vocab file is dumped at {}".format(to_tar_fn) )


if __name__ == '__main__':
    fns = [ 'chitchat.n2m.txt' ]
    load_all_dump_vocab(fns)
