import copy, re

# tag_pat = re.compile(r'[<\[]([^>\]]*)[>\]]([^<\[]*)[<\[]\/([^>\]]*)[>\]]') # <> or []
tag_pat = re.compile(r'[<]([^>]*)[>]([^<]*)[<]\/([^>]*)[>]') # <> only


def parse_a_line(line, with_2nd_level=False):
    # Read a sentence and return parsed data
    # input  : <BN:TRS>301</BN:TRS> <TT:TRS>����</TT:TRS> <INFO_TYPE:INFO_TYPE>�뼱</INFO_TYPE:INFO_TYPE> �˷���
    #        --> < 1st_level_tag | 2nd_level_Tag >
    # output : python object
        
    def bio( input_as_list , tag):
        result = []
        for idx, c in enumerate( input_as_list ):
            if idx == 0 :
                result.append( 'B-' + tag )
            else:
                result.append( 'I-' + tag )

        assert len(input_as_list) == len( result )
        return result

    from itertools import repeat
    def pad(input, maxlen):
        # list input �� maxlen ��ŭ padding �Ѵ�
        # �� ���� None ���� ä��
        input.extend(repeat(None, maxlen - len(input)))
        return input

    sentence = line

    ## ADD <s> or </s>
    #sentence = u"\u0001" + sentence + u"\u0002"

    tags     = list( copy.deepcopy( sentence ) ) # as character
    raw      = list( copy.deepcopy( sentence ) ) # as character
    level_2s = list( copy.deepcopy( sentence ) ) # as character

    sent_string = ''.join( tags )
    tags_info = tag_pat.findall( sent_string )                                      
    positions = [ m for m in re.finditer(tag_pat, sent_string)] # --> [0, 10]

    for position in positions:
        start, end = position.start(0), position.end(0)
        lex        = position.groups()[1]
        tag        = position.groups()[0]

        if ':' in tag:
            tag, level_2 = tag.split(':')
        else:
            level_2 = 'O'

        tags[start:end]      = pad( bio(list(lex), tag) , end-start )
        level_2s[start:end]  = pad( bio(list(lex), level_2) , end-start )
        raw[start:end]       = pad( list(lex), end-start)

    raw      = [ x for x in raw if x ]
    tags     = [ x for x in tags if x ]
    level_2s = [ x for x in level_2s if x ]

    # O processing
    for idx, c in enumerate(tags):
        if not ( c.startswith('B-') or c.startswith('I-') ) : tags[idx] = 'O'
        if tags[idx] in ['B-O', 'I-O']: tags[idx] = 'O'

    for idx, c in enumerate(level_2s):
        if not ( c.startswith('B-') or c.startswith('I-') ) : level_2s[idx] = 'O'
        if level_2s[idx] in ['B-O', 'I-O']: level_2s[idx] = 'O'

    assert len(raw) == len(tags)

    if with_2nd_level: return raw, tags, level_2s
    else:              return raw, tags




