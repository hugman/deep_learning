""" POS Tagger - Recurrent Neural Network tutorial Code

  This script shows how to build RNN sequence learning code
    - sequence wise loss
    - sequence wise update 

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
      sentence_wise = True  --> return list of tokens as result. PADDING is included when needed 
  """

  _sentences_in_data = []
  _labels_in_data    = []
  _weights_in_data   = []
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
      _label  = [ l_idx for l_idx in label ] 
      _weight = [ 1.0 for i in range( len(label) ) ]

      # padding to right side
      num_pad = num_steps - len(_label)
      for i in range(num_pad): _label.append( 0 ) # 0 = PADDING ID
      _labels_in_data.append( _label )

      # padding label should be weighted as 0.0
      #for i in range(num_pad): _weight.append( 0.0 )  # --> if you want to ignore pad label to calculate loss. 0.0 weight for pad 
      for i in range(num_pad): _weight.append( 1.0 )   # --> for this tutorial, just train pad as well as other symbols to get high precision since this scripts use in_top_k api to calculate precision.
      _weights_in_data.append( _weight )

    # _sentences_in_data : a list of list  - word index   shape = num_examples x num_steps
    # _labels_in_data    : a list of list  - label index  shape = num_examples x num_steps
    # _weights_in_data    : a list of list - weight       shape = num_examples x num_steps
    return _labels_in_data, _sentences_in_data, _weights_in_data

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
      np_item = np.array( np.reshape(ar, (num_ex, emb_dim) ), dtype=np.float32)
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
from tf_models.rnn_cell import GRUCell, MultiRNNCell, LSTMCell
def build_rnn_model(image_inputs, input_dim, output_dim, hidden_1=128, hidden_2=32):
  
  # X --> rnn hidden 1
  outputs   = [] # a list of output  -- len(outputs) = num_steps
  states    = [] # a list of states  -- len(states)  = num_steps
  batch_size = tf.shape(image_inputs[0])[0]
  
  with tf.name_scope('rnn_1') as scope:
    #rnn_cell  = LSTMCell(hidden_1, 50)
    rnn_cell  = GRUCell(hidden_1)
    state     = rnn_cell.zero_state(batch_size, tf.float32)

    # connect rnn cell over timeline
    num_steps = len( image_inputs )
    for t in range( num_steps ):
      if t > 0: tf.get_variable_scope().reuse_variables()
      hidden_out, state = rnn_cell(image_inputs[t], state) 

      with tf.name_scope('softmax'):
        W       = tf.get_variable("weights", [hidden_1, output_dim], initializer=tf.truncated_normal_initializer(stddev=1.0/ math.sqrt(float(hidden_1)) ) )
        b       = tf.get_variable("biases",  [output_dim])
        logits  = tf.matmul(hidden_out, W) + b

      outputs.append( logits )  # [batch_size x num_class]
      states.append( state )    # [batch_size x hidden_1]

    return outputs

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
def check_performance(sess, eval_op, pl_image_inputs, pl_label_inputs, pl_weight_inputs, stepwise_sents, stepwise_labels, stepwise_weights, batch_size):
  """ check precision performance """
  b_start_index = 0
  all_num_corrects = []
  all_num_examples = []

  num_example = stepwise_sents[0].shape[0]
  num_steps   = len( stepwise_sents )

  while True:
    b_end_index = b_start_index + batch_size
    if b_end_index > num_example: break
        
    feed_dict = {}
    for t in xrange(num_steps):
      batch_images  = stepwise_sents[t][b_start_index:b_end_index]
      batch_labels  = stepwise_labels[t][b_start_index:b_end_index]
      batch_weights = stepwise_weights[t][b_start_index:b_end_index]

      feed_dict[ pl_image_inputs[t].name ]  = batch_images
      feed_dict[ pl_label_inputs[t].name ]  = batch_labels
      feed_dict[ pl_weight_inputs[t].name ] = batch_weights

      all_num_examples.append( batch_images.shape[0] )

    num_c = sess.run(eval_op, feed_dict=feed_dict) # how many correct answers
    all_num_corrects.append( num_c )
    b_start_index = b_end_index
        
  total_num_examples = np.sum(all_num_examples)  
  total_num_corrects = np.sum(all_num_corrects)
  return total_num_examples, total_num_corrects
      
def rnn_eval_and_dump_result(sess, outputs, pl_image_inputs, stepwise_sents, vocab, _sents, labels, batch_size):
  """ evaluate the data and dump result for using conlleval.pl to get f-measure """
  f = open('__out.txt', 'w')
  b_start_index = 0
  word_2_idx, idx_2_word, label_2_idx, idx_2_label = vocab
  num_example    = len( _sents )
  num_steps      = len( stepwise_sents )
  all_num_corrects = []
  all_num_examples = []

  step_preds = []
  for t in xrange(num_steps): step_preds.append( [] )

  while True:
    b_end_index = b_start_index + batch_size
    if b_end_index > num_example: break
        
    feed_dict = {}
    for t in xrange(num_steps):
      batch_images  = stepwise_sents[t][b_start_index:b_end_index]
      feed_dict[ pl_image_inputs[t].name ]  = batch_images

    step_logits = sess.run( outputs , feed_dict ) # a list of [batch_size, num_class]

    for t in xrange(num_steps):
      batch_pred_label_ids = np.argmax( step_logits[t], axis=1 ) # [batch_size]
      batch_pred_label_sym = [ idx_2_label[l] for l in batch_pred_label_ids ]
      step_preds[t] += batch_pred_label_sym

    b_start_index = b_end_index

  # after all testing
  # shape of _sents  = [ num_ex , num_steps ]  ex) (706, 7)
  num_ex = len( step_preds[0] )


  for ex in range( num_ex ):
    for step in range( num_steps ):
      w      = idx_2_word[ _sents[ex][step] ]
      ref_l  = idx_2_label[ int(labels[step][ex]) ]
      pred_l = step_preds[step][ex]
      item   = [ w, '_', ref_l, pred_l ]

      # ignore padding
      if w == PAD : continue 
      print >> f, u" ".join(item)
    print >> f
  f.close()


import sys
def stop_here(): sys.exit()

from tf_models.seq2seq import sequence_loss
def trainer():
  # In this script
  #
  # show several advanced technique to train rnn 
  #
  #  + how to decrease learning rate with decay factor
  #  + how to change optimizer during training
  # 
  num_steps = 7

  _emb = load_emb_from_pkl()
  _labels, _sents, vocab, emb = load_data(emb=_emb, sent_len=num_steps, margin=4)
  word_2_idx, idx_2_word, label_2_idx, idx_2_label = vocab

  # reshape data 
  # sentence wise data processing
  # shape of _lables  = [ num_ex , num_steps ]  ex) (706, 7)
  # shape of _sents   = [ num_ex , num_steps ]  ex) (706, 7)
  # shape of _weights = [ num_ex , num_steps ]  ex) (706, 7)

  _labels, _sents, _weights = reshape_data(_labels, _sents, sentence_wise=True, num_steps=num_steps)

  # stepwise_sents should be = a list of [ num_ex, emb_dim ] 
  stepwise_sents    = vectorize(_sents, emb, vocab, sentence_wise=True) # vectorize as numpy obj
  
  # like sent, change the shape of _labels
  # stepwise_labels = a list of [ num_ex ] 
  _labels  = np.split( np.array(_labels),  num_steps, axis=1 )  # [ num_steps, num_ex ]
  _weights = np.split( np.array(_weights), num_steps, axis=1 )  # [ num_steps, num_ex ]
  stepwise_labels  = [ np.array(np.reshape(i, (-1) ), dtype=np.int32)   for i in _labels ]
  stepwise_weights = [ np.array(np.reshape(i, (-1) ), dtype=np.float32) for i in _weights ]

  emb_dim        = len(emb['a'])  # there should be 'a' in the emb dict
  num_class      = len( label_2_idx.keys() )
  num_example    = len( _sents )
  
  INPUT_DIM  = emb_dim
  OUTPUT_DIM = num_class

  with tf.Graph().as_default():
    batch_size    = 10
    learning_rate_decay_factor = 0.5
    init_learning_rate = 0.5
    
    learning_rate = tf.Variable(float(init_learning_rate), trainable=False)
    init_learnint_rate_op  = learning_rate.assign( 0.01 )
    learning_rate_decay_op = learning_rate.assign( learning_rate * learning_rate_decay_factor)
 
    # now change the placeholder to handle step_wise data
    pl_image_inputs  = []
    pl_label_inputs  = []
    pl_weight_inputs = []
    for t in xrange(num_steps):  
      pl_image_inputs.append(  tf.placeholder(tf.float32, shape=[None,  emb_dim], name="pl_image{}".format(t))) # image
      pl_label_inputs.append(  tf.placeholder(tf.int32,   shape=[None],           name="pl_label{}".format(t))) # label
      pl_weight_inputs.append( tf.placeholder(tf.float32, shape=[None],           name="pl_weight{}".format(t))) # weight

    # Model Build
    # outputs = a list of [batch_size, hidden_1] --> len(outputs) = num_steps
    outputs = build_rnn_model(pl_image_inputs, INPUT_DIM, OUTPUT_DIM, hidden_1=70, hidden_2=128)

    # Loss for update
    loss = sequence_loss(outputs, pl_label_inputs, pl_weight_inputs, num_class)
    tf.scalar_summary('sequence_loss', loss) # register loss to monitor

    # Parameter Update operator
    opt1 = tf.train.GradientDescentOptimizer(0.01)
    opt2 = tf.train.AdagradOptimizer(learning_rate)
    opt3 = tf.train.MomentumOptimizer(learning_rate, momentum=0.01)
    opt4 = tf.train.AdamOptimizer(learning_rate)
    opt5 = tf.train.RMSPropOptimizer(learning_rate, decay=0.001)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = opt2 # start with faster optimizer 
    
    train_op  = optimizer.minimize(loss)
    train_op2 = opt1.minimize(loss)
    
    # Evaluation Operator  (sequence wise)
    num_corrects_list = []
    for t in xrange(num_steps):
      step_num_corrects = tf.nn.in_top_k(outputs[t], pl_label_inputs[t], 1, name='num_correct_at_{}'.format(t))  
      num_corrects_list.append( step_num_corrects )
    eval_op = tf.reduce_sum( tf.cast(tf.pack( num_corrects_list ), tf.int32), name='num_corrects')

    # Create summary op for monitoring
    tf_prec    = tf.Variable(0.0, trainable=False) # variable for precision monitoring
    tf.scalar_summary('precision', tf_prec) # register precision to monitor
    summary_op = tf.merge_all_summaries()

    # Session
    sess = tf.Session()

    # Create writer op for monitoring
    monitor_dir = '/local/log-tensorboard'
    summary_writer = tf.train.SummaryWriter(monitor_dir, graph_def=sess.graph_def) # register writer

    # Init all variable
    init = tf.initialize_all_variables()
    sess.run(init)

    show_model_params()

    # --------------------------------------------------------------------#
    # train 
    # RNN training should be timestep wise
    # 

    step = 0
    check_losses = []
    previous_losses = []
    for epoch in range(1000):
      b_start_index = 0
      # batch processing
      while True:
        b_end_index = b_start_index + batch_size
        if b_end_index > num_example: break
        
        feed_dict = {}
        for t in xrange(num_steps):
          batch_images  = stepwise_sents[t][b_start_index:b_end_index]
          batch_labels  = stepwise_labels[t][b_start_index:b_end_index]
          batch_weights = stepwise_weights[t][b_start_index:b_end_index]

          start_time = time.time()

          feed_dict[ pl_image_inputs[t].name ]  = batch_images
          feed_dict[ pl_label_inputs[t].name ]  = batch_labels
          feed_dict[ pl_weight_inputs[t].name ] = batch_weights
        
        # update overall time steps
        # using tensorflow, we can even change the optimizer during training
        """
        if epoch > 50 and train_op.name != 'GradientDescent':
          sess.run(init_learnint_rate_op) # reset learning rate for new optimizer 
          train_op = train_op2  # change optimizer
          previous_losses.append( check_mean_loss ) # reset history
        """
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        check_losses.append( loss_value )
        duration = time.time() - start_time
        # monitor loss per step 
        if step % 50 == 0:
            check_mean_loss = np.mean(check_losses)
            # how to change learning rate during training with decay factor
            if len(previous_losses) > 2 and check_mean_loss > np.max(previous_losses[-3:]):
              changed_flag = '<-- LR changed'
              sess.run(learning_rate_decay_op) # decay learnig rate
            else:
              changed_flag = ''

            previous_losses.append(check_mean_loss)

            current_lr = sess.run(learning_rate)
            print train_op.name, ' Epoch %d - Step %d: loss = %.2f with lr = %.7f (%.3f sec)' % (epoch, step, check_mean_loss, current_lr, duration), changed_flag
            check_losses = [] # init

            # monitoring
            summary_str = sess.run(summary_op, feed_dict=feed_dict)  # get summary info from summary op
            summary_writer.add_summary(summary_str, step)            # add summary

        step += 1
        
        b_start_index = b_end_index

      # every n epoch, check performance
      if epoch % 10 == 0 :
        total_num_examples, total_num_corrects = check_performance(sess, eval_op, pl_image_inputs, pl_label_inputs, pl_weight_inputs, stepwise_sents, stepwise_labels, stepwise_weights, batch_size)
        prec = float(total_num_corrects) / float(total_num_examples)
        print '\tNum examples: %d  Num correct: %d  Precision @ 1: %0.04f' % ( total_num_examples, total_num_corrects, prec )
        rnn_eval_and_dump_result(sess, outputs, pl_image_inputs, stepwise_sents, vocab, _sents, _labels, batch_size)

        # monitoring
        sess.run( tf_prec.assign( prec ) )
                                
def main(_):
    trainer()

if __name__ == '__main__':
    tf.app.run()
