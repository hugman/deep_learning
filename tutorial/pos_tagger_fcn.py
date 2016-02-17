""" POS Tagger - Fully Connected Network tutorial Code

  This script handles a word as a image ignoring sentence boundary. 

  Author : Sang Keun Jung (2016)

"""

import time
import tensorflow.python.platform
import numpy as np
import tensorflow as tf

WIDTH       = 28
HEIGHT      = 28

import cPickle as pkl
def load_emb_from_pkl():
  """ Load embedding data from pkl file
        python dictionary obj file :  dict["string"] = a list of floats
  """
  print("loading embedding ...")
  emb_f = open('./data/emb.pkl', 'r')
  emb   = pkl.load(emb_f)
  return emb
 
PAD = 'PADDING' # id 0 as convention
UNK = 'UNKNOWN' # id 1 as convention

import operator
def load_data(emb=None, sent_len=7, margin=2):
  """ filter sentences where the length of the sentence is less than sent_len and bigger than sent_len-margin
      ex) sent_len=7 and margin=3 --> sentence length 5, 6, 7 are selected. others are filtered out.  

  """
  fn = './data/wsj.all.data'  # just for tutorial purpose, all = train + dev + test data
  print "Data loading ...", fn
  f = open(fn, 'r')
  sent = []
  data = []
  for line in f:
    if line == '\n':
      data.append( sent ) 
      sent = []
      continue

    line = line.rstrip()
    try   :  word, tag = line.split()
    except: continue # if irregular format, skip

    if word not in ['PADDING', 'UNKNOWN']: word = word.lower()
    sent.append( (word, tag) )
  f.close()
  
  sentences_in_data = []
  labels_in_data    = [] 
  for sent in data:
    sentence = []
    labels   = []
    for word, tag in sent:
      sentence.append( word ) 
      labels.append( tag )

    l_bound = sent_len - margin
    r_bound = sent_len
    if len(sentence) < l_bound or len(sentence) > r_bound: continue

    sentences_in_data.append( sentence )
    labels_in_data.append( labels )

  # unique words
  num_total_words = 0
  words = {}
  for sent in sentences_in_data:
    for w in sent:
      if w not in words : words[w] = 0
      words[w] += 1
      num_total_words += 1

  # unique labels
  num_total_labels = 0 
  labels = {}
  for _labels in labels_in_data:
    for l in _labels:
      if l not in labels: labels[l] = 0
      labels[l] += 1
      num_total_labels += 1

  # sort by frequency 
  words  = sorted(words.items(), key=operator.itemgetter(1), reverse=True) 
  labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True) 

  ### NOTICE ###
  # For tutorial puspose, word, label and emb are limited to target documents ( to speed up )
  # documents = set of sentence (where less than sent_len and bigger than sent_len-margin )

  # build index for words and labels
  word_2_idx  = {}; idx_2_word  = {}
  label_2_idx = {}; idx_2_label = {}

  # add padding and unknown 
  word_2_idx[PAD]  = 0; idx_2_word[0]  = PAD
  label_2_idx[PAD] = 0; idx_2_label[0] = PAD

  word_2_idx[UNK]  = 1; idx_2_word[1]  = UNK
  label_2_idx[UNK] = 1; idx_2_label[1] = UNK

  start_idx = len( [PAD, UNK] )
  for idx, (w, fr) in enumerate(words):
    word_2_idx[w] = start_idx + idx
    idx_2_word[start_idx + idx] = w
  for idx, (l, fr) in enumerate(labels):   # for tutorial purpose only.
    label_2_idx[l] = start_idx + idx
    idx_2_label[start_idx + idx] = l

  # replace symbols to index
  _sentences_in_data = []
  _labels_in_data = []
  for sent in sentences_in_data:
    _sent = [ word_2_idx[w] for w in sent ] 
    _sentences_in_data.append( _sent )

  for label in labels_in_data:
    _label = [ label_2_idx[l] for l in label ] 
    _labels_in_data.append(_label)

  # select emb dict
  new_emb = {}
  new_emb[PAD] = emb[PAD]
  new_emb[UNK] = emb[UNK]
  for w in word_2_idx.keys(): new_emb[w] = emb[w]


  print "--------------------"
  print "Data summary"
  print "--------------------"
  print "# of sentences                 : ", len(_sentences_in_data)
  print "# of words in data set         : ", num_total_words   # in here, word is just a token.. not 'word'
  print "# of labels in data set        : ", num_total_labels
  print "# of unique words in data set  : ", len(words)
  print "# of unique labels in data set : ", len(labels)
  print "Dimension of embedding dict    : ", len(new_emb['a'])
  print "# of key in emb dict           : ", len(new_emb.keys())
  print "--------------------"

  # labels_in_data    : a list of list of tag index
  # sentences_in_data : a list of list of word index
  vocab = (word_2_idx, idx_2_word, label_2_idx, idx_2_label)
  return _labels_in_data, _sentences_in_data, vocab, new_emb

def reshape_data(labels_in_data, sentences_in_data, word_as_ex=True):
  """ reshape data to handle data as tensor
      mode
         word_as_ex = True  --> convert all data to very long word level examples. ignore sentence boundary 
  """
  _sentences_in_data = []
  _labels_in_data    = []
  if word_as_ex:
    for sent in sentences_in_data:
      for w_idx in sent: _sentences_in_data.append( w_idx )
    for label in labels_in_data:
      for l_idx in label: _labels_in_data.append( l_idx )
    return _labels_in_data, _sentences_in_data

def vectorize(sents, emb, vocab, word_as_ex=True):
  """ convert index based data to dense vector data """
  word_2_idx, idx_2_word, _, _ = vocab

  if word_as_ex:
    sents = [ emb[ idx_2_word[w_idx] ] for w_idx in sents ]
    np_sents = np.array( sents ) 
    return np_sents 

def variable_init_2D(num_input, num_output):
  """Initialize weight matrix using truncated normal method
      check detail from Lecun (98) paper. 
       - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  init_tensor = tf.truncated_normal([num_input, num_output], stddev=1.0 / math.sqrt(float(num_input)))
  return init_tensor


import math
def build_model(images, input_dim, output_dim, hidden_1=128, hidden_2=32):
  """ Build neural network model """
  # X --> Hidden 1
  with tf.name_scope('hidden1') as scope:
    W      = tf.Variable( variable_init_2D(input_dim, hidden_1), name='weights')
    b      = tf.Variable( tf.zeros([hidden_1]), name='biases')
    out_h1 = tf.nn.relu(  tf.matmul(images, W) + b)
  
  # Hidden 1 --> Hidden 2
  with tf.name_scope('hidden2') as scope:
    W      = tf.Variable( variable_init_2D(hidden_1, hidden_2), name='weights')
    b      = tf.Variable( tf.zeros([hidden_2]), name='biases')
    out_h2 = tf.nn.relu(  tf.matmul(out_h1, W) + b)

  # Hidden 2 --> Y
  with tf.name_scope('softmax') as scope:
    W      = tf.Variable( variable_init_2D(hidden_2, output_dim), name='weights')
    b      = tf.Variable(tf.zeros([output_dim]), name='biases')
    logits = tf.matmul(out_h2, W) + b

  return logits


def cross_entropy_loss(logits, labels, num_class):
    """Calculates the loss from the logits and the labels. -- from google code

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
    # to 1-hot dense float vectors (that is we will have batch_size vectors,
    # each with NUM_CLASSES values, all of which are 0.0 except there will
    # be a 1.0 in the entry corresponding to the label).
    batch_size    = tf.size(labels)
    labels        = tf.expand_dims(labels, 1)
    indices       = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated      = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, num_class]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss



def show_model_params():
  print "-----------------------------"
  print "Trainable Variables"
  print "-----------------------------"
  for v in tf.trainable_variables(): print v.name
  print "-----------------------------"


def next_batch(images, labels, batch_size):
  """ Yield batch data. 
      Note that smaller batch than batch_size is ignored """
  # iterator 
  start  = 0 
  num_ex = images.shape[0]
          
  while True:
    end = start + batch_size
    if end > num_ex: break 
    yield images[start:end], labels[start:end]
    start = end

import numpy as np
def check_performance(sess, eval_op, pl_images, pl_labels, images, labels, batch_size):
  """ check precision performance """
  true_counts = []
  num_examples = 0
  for batch_images, batch_labels in next_batch(images, labels, batch_size):
    feed_dict = { 
        pl_images: batch_images,
        pl_labels: batch_labels,
    }
    true_count = sess.run(eval_op, feed_dict=feed_dict) 
    true_counts.append( true_count )
    num_examples += batch_size

  return num_examples, true_counts

def eval_and_dump_result(sess, pred_labels, pl_images, pl_labels, images, labels, batch_size, vocab, _sents, word_as_ex=True):
  """ evaluate the data and dump result for using conlleval.pl to get f-measure """
  f = open('__out.txt', 'w')
  word_2_idx, idx_2_word, label_2_idx, idx_2_label = vocab
  
  if word_as_ex:
    batch_idx = 0
    global_idx = 0
    for batch_images, batch_labels in next_batch(images, labels, batch_size):
      feed_dict = { 
          pl_images: batch_images,
          pl_labels: batch_labels,
      }
    
      batch_pred_labels = sess.run(pred_labels, feed_dict=feed_dict) 
      # dump result ignoring setence boundary 
      for idx, ref_label in enumerate(batch_labels):
        global_idx = batch_idx * batch_size + idx
        item = [ idx_2_word[ _sents[global_idx]], '_', idx_2_label[ref_label], idx_2_label[batch_pred_labels[idx]] ]
        print >> f, u" ".join(item)
      print >> f
      batch_idx += 1
    f.close()


def show_data(labels, sents, index=0, show_num=False):
  # show data (console)
  label = labels[index]
  sent  = sents[index]

  lst = []
  for idx, s in enumerate(sent):
    lst.append( "{}/{}".format(s, label[idx]) )

  print " ".join(lst)

import sys
def stop_here(): sys.exit()

def trainer():
  # In this script
  #   ! we handle text as just like image  (same model as mnist)
  # 

  # Data Preparation - Block 0
  _emb = load_emb_from_pkl()
  _labels, _sents, vocab, emb = load_data(emb=_emb, sent_len=7, margin=4)
  word_2_idx, idx_2_word, label_2_idx, idx_2_label = vocab

  # reshape data 
  # a token as a single example just like MNIST image 
  _labels, _sents = reshape_data(_labels, _sents, word_as_ex=True)
  sents           = vectorize(_sents, emb, vocab) # vectorize as numpy obj

  #show_data(labels, sents, index=4); stop_here()  # toggle show_num=True and False 
  
  # for tutorial pupose only
  #     - image = token
  #     - label = pos tag
  images, labels = sents, _labels
  emb_dim        = len(emb['a'])  # there should be 'a' in the emb dict
  num_class      = len( label_2_idx.keys() )
  
  INPUT_DIM  = emb_dim
  OUTPUT_DIM = num_class

  with tf.Graph().as_default():
    batch_size    = 100
    learning_rate = 0.01
         
    pl_images = tf.placeholder(tf.float32, shape=(batch_size, INPUT_DIM), name='pl_image')
    pl_labels = tf.placeholder(tf.int32,   shape=(batch_size), name='pl_label')

    # Model Build - Block 1
    logits = build_model(pl_images, INPUT_DIM, OUTPUT_DIM, hidden_1=128, hidden_2=70)
        
    # Loss for update - Block 2
    loss = cross_entropy_loss(logits, pl_labels, num_class)

    # Parameter Update operator - Block 3
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op  = optimizer.minimize(loss)

    # Evaluation Operator
    num_corrects = tf.nn.in_top_k(logits, pl_labels, 1)  
    eval_op      = tf.reduce_sum(tf.cast(num_corrects, tf.int32))

    # Result Dumping Operator
    pred_labels  = tf.argmax(logits, dimension=1)

    # Session
    sess = tf.Session()

    # Init all variable
    init = tf.initialize_all_variables()
    sess.run(init)

    show_model_params()

    # --------------------------------------------------------------------#
    # train - Block 4
    step = 0
    check_losses = []
    for epoch in range(1000):
      for batch_images, batch_labels in next_batch(images, labels, batch_size):
        start_time = time.time()
        feed_dict = { 
            pl_images: batch_images,
            pl_labels: batch_labels,
        }

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        check_losses.append( loss_value )
        duration = time.time() - start_time
          
        # monitor loss per step 
        if step % 50 == 0:
            print 'Epoch %d - Step %d: loss = %.2f (%.3f sec)' % (epoch, step, np.mean(check_losses), duration)
            check_losses = []
        step += 1

      # every n epoch, check performance
      if epoch % 10 == 0 :
        total_num_examples, b_true_counts = check_performance(sess, eval_op, pl_images, pl_labels, images, labels, batch_size)
        total_true_counts = np.sum(b_true_counts)
        prec = float(total_true_counts) / float(total_num_examples) 
        print '\tNum examples: %d  Num correct: %d  Precision @ 1: %0.04f' % ( total_num_examples, total_true_counts, prec )
        eval_and_dump_result(sess, pred_labels, pl_images, pl_labels, images, labels, batch_size, vocab, _sents)
               
def main(_):
    trainer()

if __name__ == '__main__':
    tf.app.run()
