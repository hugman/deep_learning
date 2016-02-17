""" Mnist tutorial Code with Monitoring features -- with Tensorboard.

  Author : Sang Keun Jung (2016)
           hugmanskj@gmail.com

"""

import time
import tensorflow.python.platform
import numpy as np
import tensorflow as tf


import cPickle as pkl
def load_from_pkl():
  """ Load data from pkl file
        Pkl file contains numpy object 
        image : shape [num_examples, 784]  - float
        label : shape [num_exmaples]       - int
  """
  print("loading ...")

  np_label_file = open('./data/np_label_file.10k.pkl', 'rb')
  labels  = pkl.load( np_label_file )

  np_image_file = open('./data/np_image_file.10k.pkl', 'rb')
  images  = pkl.load( np_image_file )

  num_examples = images.shape[0]

  print "--------------------"
  print "Data summary"
  print "--------------------"
  print "# of examples : ", num_examples
  print "Label shape   : ", labels.shape
  print "Image shape   : ", images.shape
  print "--------------------"

  return labels, images, num_examples


def variable_init_2D(num_input, num_output):
  """Initialize weight matrix using truncated normal method
      check detail from Lecun (98) paper. 
       - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  init_tensor = tf.truncated_normal([num_input, num_output], stddev=1.0 / math.sqrt(float(num_input)))
  return init_tensor

import math
def build_model(images, input_dim, output_dim, hidden_1=128, hidden_2=32):
  """ Build neural network model 
  """
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
    """Calculates the loss from the logits and the labels. -- from google tutorial code

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


def show_data(labels, images, index=0, show_num=False):
  # show data (console)

  label = labels[index]
  image = images[index]

  image = np.reshape( image, (28,28) ) # to visualize, reshpae 28x28
  print "Label = ", label
  for r in range(28):
    for c in range(28):
      if show_num :
        mark = u'1' if image[r][c] > 0.0 else '0'
      else:
        mark = u'*' if image[r][c] > 0.0 else ' '
      print mark,
    print 

  print label
  
import sys
def stop_here(): sys.exit()

def trainer():

  # data preapration  - Block 0
  labels, images, num_examples = load_from_pkl()
  #show_data(labels, images, index=4, show_num=True); stop_here()  # toggle show_num=True and False 

  WIDTH     = 28
  HEIGHT    = 28
  num_class = 10 # number of digit 

  INPUT_DIM  = WIDTH * HEIGHT  
  OUTPUT_DIM = num_class  
  with tf.Graph().as_default():
    batch_size    = 100
    learning_rate = 0.01
         
    pl_images = tf.placeholder(tf.float32, shape=(batch_size, INPUT_DIM), name='pl_image')
    pl_labels = tf.placeholder(tf.int32,   shape=(batch_size), name='pl_label')

    # Model Build - Block 1
    logits = build_model(pl_images, INPUT_DIM, OUTPUT_DIM)
        
    # Loss for update - Block 2
    loss = cross_entropy_loss(logits, pl_labels, num_class)
    tf.scalar_summary('cross_entropy', loss) # register loss to monitor

    # Parameter Update operator - Block 3
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op  = optimizer.minimize(loss)

    # Evaluation Operator
    num_corrects = tf.nn.in_top_k(logits, pl_labels, 1)  
    eval_op      = tf.reduce_sum(tf.cast(num_corrects, tf.int32)) # how many 

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
            check_losses = [] # init

            # monitoring
            summary_str = sess.run(summary_op, feed_dict=feed_dict)  # get summary info from summary op
            summary_writer.add_summary(summary_str, step)            # add summary
        step += 1

      # every n epoch, check performance
      if epoch % 10 == 0 :
        #saver.save(sess, monitor_dir, global_step=step)
        b_num_examples, b_true_counts = check_performance(sess, eval_op, pl_images, pl_labels, images, labels, batch_size)
        all_num_examples  =  b_num_examples
        total_true_counts = np.sum(b_true_counts)
        prec              = float(total_true_counts) / float(all_num_examples)
        print '\tNum examples: %d  Num correct: %d  Precision @ 1: %0.04f' % ( all_num_examples, total_true_counts, prec )

        # monitoring
        sess.run( tf_prec.assign( prec ) )
               

def main(_):
    trainer()

if __name__ == '__main__':
    tf.app.run()
