# english pos tagger data prepare 
# 

import os
def load_from_rawfile(fn):
  print "Data loading ...", fn
  f = open(fn, 'r')
  sent = []
  data = []
  for line in f:
    if line == '\n':
      data.append( sent ) 
      text = [ x[0] for x in sent ] 
      sent = []
      continue

    line = line.rstrip()
    try   :  word, tag = line.split()
    except: continue # if irregular format, skip

    if word not in ['PADDING', 'UNKNOWN']:
      word = word.lower()

    sent.append( (word, tag) )
  f.close()

  # unique words
  words = {}
  for sent in data:
    for w, t in sent:
      if w not in words : words[w] = 0
      words[w] += 1
  return data, words


def load_embedding_dict(word_list_fn, emb_fn):
  # load embedding table from file
  print("Embedding file loading ...")

  word_2_idx = {}
  idx_2_word = {}
  word_f = open(word_list_fn, 'r')
  index = 0
  for line in word_f:
    line = line.rstrip()
    word_2_idx[line] = index
    idx_2_word[index] = line
    index += 1
  word_f.close()

  emb = {}
  emb_f = open(emb_fn, 'r')
  index = 0
  for line in emb_f:
    line = line.rstrip()
    vector_s = line.split()
    vector   = [ float(x) for x in vector_s ]
    emb[ idx_2_word[index] ] = vector
    index += 1

  return emb

emb = load_embedding_dict('words.lst', 'embeddings.txt')

dev_data,   dev_words   = load_from_rawfile('wsj.dev.data')
test_data,  test_words  = load_from_rawfile('wsj.test.data')
train_data, train_words = load_from_rawfile('wsj.train.data')

all_words = train_words.keys() + dev_words.keys() + test_words.keys()

# filter by appeared words only 
filtered_emb = {}
nc  = 0
ncw = []
for w in all_words:
  if w in emb:
    filtered_emb[w] = emb[w]
  else:
    nc += 1
    filtered_emb[w] = emb['UNKNOWN']
    ncw.append( w )  # not in embedding dict

# add UNKNOWN, PADDING to the emb dictionary
filtered_emb['UNKNOWN'] = emb['UNKNOWN']
filtered_emb['PADDING'] = emb['PADDING']

import cPickle as pkl
of = open('emb.pkl', 'w')
pkl.dump(filtered_emb, of)
of.close()

## summary ##
print "# of words : ", len(all_words)
print "UNKNOWN :", filtered_emb['UNKNOWN']
print "PADDING :", filtered_emb['PADDING']
