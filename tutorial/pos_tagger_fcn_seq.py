""" POS Tagger - Fully Connected Network tutorial Code

  This script handles sentence wise model build, loss and evaluation

  Author : Sang Keun Jung (2016)

"""

import time
import tensorflow.python.platform
import numpy as np
import tensorflow as tf

PAD = 'PADDING' # id 0 as convention
UNK = 'UNKNOWN' # id 1 as convention

import cPickle as pkl
def load_emb_from_pkl():
  """ Load embedding data from pkl file
         dict["string"] = a list of floats
  """
  print("loading embedding ...")
  emb_f = open('./data/emb.pkl', 'r')
  emb   = pkl.load(emb_f)
  return emb
 

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

def reshape_data(labels_in_data, sentences_in_data, word_as_ex=True, sentence_wise=False, num_steps=0):
  """ reshape data to handle data as tensor
   mode
      word_as_ex    = True  --> convert all data to very long word level examples. ignore sentence boundary 
      sentence_wise = True  --> return list of tokens as result. PADDING is when needed 
  """

  _sentences_in_data = []
  _labels_in_data    = []
  if word_as_ex and sentence_wise == False:
    for sent in sentences_in_data:
      for w_idx in sent: _sentences_in_data.append( w_idx )

    for label in labels_in_data:
      for l_idx in label: _labels_in_data.append( l_idx )

    # _sentences_in_data : a list of word index
    # _labels_in_data    : a list of label index
    return _labels_in_data, _sentences_in_data

  if sentence_wise:
    for s_idx, sent in enumerate(sentences_in_data):
      _sent = [ w_idx for w_idx in sent ] 
      
      # padding to right side
      num_pad = num_steps - len(_sent)
      for i in range(num_pad): _sent.append( 0 ) # 0 = PADDING ID
      _sentences_in_data.append( _sent )

    for label in labels_in_data:
      _label = [ l_idx for l_idx in label ] 

      # padding to right side
      num_pad = num_steps - len(_label)
      for i in range(num_pad): _label.append( 0 ) # 0 = PADDING ID
      _labels_in_data.append( _label )


    # _sentences_in_data : a list of list - word index   shape = [num_examples , num_steps]
    # _labels_in_data    : a list of list - label index  shape = [num_examples , num_steps]
    return _labels_in_data, _sentences_in_data

def vectorize(sents, emb, vocab, word_as_ex=True, sentence_wise=False):
  word_2_idx, idx_2_word, _, _ = vocab

  if word_as_ex and sentence_wise == False:
    sents = [ emb[ idx_2_word[w_idx] ] for w_idx in sents ]
    np_sents = np.array( sents ) 
    return np_sents 

  if sentence_wise:
    # sents = [ num_ex x num_steps ]
    _sents = []
    for sent in sents:
      _sent = []
      for w_idx in sent: 
        _sent.append( emb[ idx_2_word[w_idx] ] )

      _sents.append( _sent ) 

    # now sents shape should be = [ num_ex, num_steps, emb_dim ] 
    np_sents = np.array( _sents ) 
    
    num_ex    = np_sents.shape[0]
    num_steps = np_sents.shape[1]
    emb_dim   = np_sents.shape[2]

    # turn np_sents to a list of item where len(list) = num_steps,  item shape = [num_ex, emb_dim]
    step_list = np.split(np_sents, num_steps, axis=1)

    result = []
    for ar in step_list:
      np_item = np.reshape(ar, (num_ex, emb_dim) )
      result.append( np_item ) 

    return result

def variable_init_2D(num_input, num_output):
  """Initialize weight matrix using truncated normal method
      check detail from Lecun (98) paper. 
       - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  init_tensor = tf.truncated_normal([num_input, num_output], stddev=1.0 / math.sqrt(float(num_input)))
  return init_tensor


import math
def build_model(images, input_dim, output_dim, hidden_1=128, hidden_2=32):
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

def eval_and_dump_result(sess, pred_labels, pl_images, pl_labels, images, labels, batch_size, vocab, _sents, word_as_ex=True, sentence_wise=False):
  """ evaluate the data and dump result for using conlleval.pl to get f-measure """
  f = open('__out.txt', 'w')
  word_2_idx, idx_2_word, label_2_idx, idx_2_label = vocab
  
  if word_as_ex and sentence_wise == False:
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

  if sentence_wise:
    batch_idx = 0
    global_idx = 0
    num_step = len(_sents[0])

    all_pred_labels = []
    for step in range(num_step):
      step_images = images[step]   # [num_ex , emb_dim]
      step_labels = labels[step]   # [num_ex]

      step_batch_pred_labels = []
      for batch_images, batch_labels in next_batch(step_images, step_labels, batch_size):
        feed_dict = { 
            pl_images: batch_images,
            pl_labels: batch_labels,
        }
        batch_pred_labels = sess.run(pred_labels, feed_dict=feed_dict) 

        for p in batch_pred_labels:
          step_batch_pred_labels.append( int(p) )

      all_pred_labels.append( step_batch_pred_labels )

    _result = np.transpose( np.array(all_pred_labels) )  # shape = (num_ex, num_steps)
    
    num_ex   = _result.shape[0]
    num_step = _result.shape[1]
    
    # shape of _sents  = [ num_ex , num_steps ]  ex) (706, 7)
    for ex in range( num_ex ):
      for step in range( num_step ):
        w      = idx_2_word[ _sents[ex][step] ]
        ref_l  = idx_2_label[ labels[step][ ex ] ]
        pred_l = idx_2_label[ _result[ex][step] ]
        item   = [ w, '_', ref_l, pred_l ]

        # ignore padding
        if w == PAD : continue 
        print >> f, u" ".join(item)
      print >> f
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
  #   ! using same model as pos_tagger_fcn.py
  #   ! but with different update methods
  #

  # Data preparation - Block 0 
  num_steps = 7
  _emb = load_emb_from_pkl()
  _labels, _sents, vocab, emb = load_data(emb=_emb, sent_len=num_steps, margin=4)
  word_2_idx, idx_2_word, label_2_idx, idx_2_label = vocab

  # reshape data 
  # sentence wise data processing
  # shape of _lables = [ num_ex , num_steps ]  ex) (706, 7)
  # shape of _sents  = [ num_ex , num_steps ]  ex) (706, 7)
  _labels, _sents = reshape_data(_labels, _sents, sentence_wise=True, num_steps=num_steps)

  # stepwise_sents should be = a list of [ num_ex, emb_dim ] 
  stepwise_sents    = vectorize(_sents, emb, vocab, sentence_wise=True) # vectorize as numpy obj
  
  # like sent, change the shape of _labels
  # stepwise_labels = a list of [ num_ex ] 
  _labels = np.split( np.array(_labels), num_steps, axis=1 )
  stepwise_labels = [ np.reshape(i, (-1) ) for i in _labels ]

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
    # Now, train method should be chaned
    #    - handle stepwise data format

    step = 0
    check_losses = []
    for epoch in range(1000):
      for step in range(num_steps):
        images, labels = stepwise_sents[step], stepwise_labels[step]
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
              check_losses = [] # init
          step += 1

      # every n epoch, check performance
      if epoch % 10 == 0 :
        step_b_num_examples = []
        step_b_true_counts  = []
        for step in range(num_steps):
          images, labels = stepwise_sents[step], stepwise_labels[step]
          b_num_examples, b_true_counts = check_performance(sess, eval_op, pl_images, pl_labels, images, labels, batch_size)
          step_b_num_examples.append( b_num_examples )
          step_b_true_counts += b_true_counts

        total_num_examples = np.sum( step_b_num_examples )
        total_num_corrects = np.sum( step_b_true_counts )
        prec = float( total_num_corrects ) / float (total_num_examples )
        print '\tNum examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (total_num_examples, total_num_corrects, prec )
        eval_and_dump_result(sess, pred_labels, pl_images, pl_labels, stepwise_sents, stepwise_labels, batch_size, vocab, _sents, sentence_wise=True)
               

def main(_):
    trainer()

if __name__ == '__main__':
    tf.app.run()
