"""
    Convert json corpus format to XML like(<tag>~</tag>) form data

    Author : Sangkeun Jung
"""

import codecs, json
import os, sys

# note that the json file format is from 
# - https://github.com/krikit/annie
# - 2016년 한글 및 한국어 정보처리 학술대회 :  "2016년 엑소브레인 V2.0 & 울산대학교 말뭉치"라는 명칭의 CD로 배포
#
def convert(fn):
    xml_sents = []
    json_fn = os.path.join( os.path.dirname(__file__), fn ) 
    with codecs.open(json_fn, 'r', encoding='utf-8') as f:
        data = json.load(f) 
        sents = data['sentence']
        for sent in sents:
            text = sent['text']
            morp = sent['morp']

            if '어떠한 도발' in text:
                x = 3
            c_text = list(text)          # character level tokens

            def get_byte_index_to_char_index(c_text):
                bid_to_cid = {}
                bid = 0
                for cid, c in enumerate(c_text):
                    byte_list = bytearray( c.encode() )
                    for b in byte_list:
                        bid_to_cid[bid] = cid
                        bid += 1
                return bid_to_cid

            bid_to_cid = get_byte_index_to_char_index(c_text)

            c_begin_tags = ['' for x in text ] # character level tags
            c_end_tags   = ['' for x in text ] # character level tags

            from_here = 0 
            for ne in sent['NE']:
                # it's not strict conversion. (for simplicity, just use pattern matching)
                ne_token = ne['text']
                ne_tag   = ne['type']
                
                from_here = bid_to_cid[ morp[ ne['begin'] ]['position'] ] 

                start_pos = text.find(ne_token, from_here) 
                if start_pos > -1: 
                    end_pos   = start_pos + len(ne_token) -1
                    c_begin_tags[start_pos] = '<{}>'.format(ne_tag.upper())
                    c_end_tags[end_pos]     = '</{}>'.format(ne_tag.upper())
                else:
                    x = 3

            # merge
            m_chars = []
            for b_tag, char, e_tag in zip(c_begin_tags, c_text, c_end_tags):
                m_chars.append( '{}{}{}'.format(b_tag, char, e_tag) ) 
            m_text = "".join( m_chars )
            xml_sents.append( m_text ) 

    # to dump xml file
    out_fn = os.path.join( os.path.dirname(__file__), 'ner.xml.txt') 
    with codecs.open(out_fn, 'w', encoding='utf-8') as f:
        for sent in xml_sents:
            print(sent, file=f)

    print("XML like file is dumped at {}".format(out_fn))


    

if __name__ == '__main__':
    fn = '2016klp_1000Sentences.json'
    convert(fn)
