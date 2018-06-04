"""
    Convert XML like(<tag>~</tag>) text data to N2N data format

    Author : Sangkeun Jung
"""
import os,sys, codecs

# add common to path
from pathlib import Path

common_path = str(Path( os.path.abspath(__file__) ).parent.parent.parent)
sys.path.append( common_path )

from common.data_format.n2n import parse_a_line

def convert(fn):
    xml_fn = os.path.join( os.path.dirname(__file__), fn ) 
    to_fn  = os.path.join( os.path.dirname(__file__), 'ner.n2n.txt' ) 

    with codecs.open(to_fn, 'w', encoding='utf-8') as of:
        with codecs.open(xml_fn, 'r', encoding='utf-8') as f:
            for line in f: 
                line = line.rstrip('\n\r')

                # convert xml like sentence to BIO tagged N2N format
                raw, tags = parse_a_line(line)

                for char, tag in zip(raw, tags):
                    print("{}\t{}".format(char, tag), file=of)
                print("-"*50, file=of)
            
    print("N2N file dumped at {}".format(to_fn))

if __name__ == '__main__':
    fn = 'ner.xml.txt'
    convert(fn)
