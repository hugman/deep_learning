"""
    Example code for timeline usage

    Author : Sangkeun Jung (hugmanskj@gmail.com)
"""

import cPickle as pkl
import os, sys
import numpy as np
import tensorflow as tf
import math

from tensorflow.python.client import timeline

# we have sampled mnist data at ./data folder
# - image file : np_image_file.10k.pkl  <-- pickled objects
# - label file : np_label_file.10k.pkl  <-- pickled objects
# 
# a image of minst is 28x28 pixel image

# loading and checking

with open('./data/np_image_file.10k.pkl') as image_f:
    images = pkl.load(image_f)

with open('./data/np_label_file.10k.pkl') as label_f:
    labels = pkl.load(label_f)    

batch_size = 100
def batch_reader(start_index):
    # read data[start_index:start_index+batch_size]
    # return [batch_size, 784], [batch_size]
    try:
        batch_images = images[start_index:start_index+batch_size]
        batch_labels = labels[start_index:start_index+batch_size]
        return np.array(batch_images), batch_labels
    except:
        return None, None

def variable_init_2D(num_input, num_output):
    """
    Initialize weight matrix using truncated normal method
      check detail from Lecun (98) paper.
       - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    init_tensor = tf.truncated_normal([num_input, num_output], stddev=1.0 / math.sqrt(float(num_input)))
    return init_tensor

def inference(batch_input):
    # batch_input : [batch_size, input_dim]  # ex) mnist case : 784
    
    # do inference & and return logits
    # return [batch_size, 10]   # 10 = [0,1,2,....,9]

    # 3 layer 
    # 784 --> 144 --> 36
    input_dim = 784
    h1_dim = 144
    h2_dim = 36
    output_dim = 10
    with tf.variable_scope("hidden_1"):
        W = tf.Variable( variable_init_2D(input_dim, h1_dim), name='weights')
        b = tf.Variable( tf.zeros([h1_dim]), name='biases')
        out_h1 = tf.nn.relu( tf.matmul(batch_input, W) + b )
    
    with tf.variable_scope("hidden_2"):
        W = tf.Variable( variable_init_2D(h1_dim, h2_dim), name='weights')
        b = tf.Variable( tf.zeros([h2_dim]), name='biases')
        out_h2 = tf.nn.relu( tf.matmul(out_h1, W) + b )
    
    with tf.variable_scope("hidden_3"):
        W = tf.Variable( variable_init_2D(h2_dim, output_dim), name='weights')
        b = tf.Variable( tf.zeros([output_dim]), name='biases')
        
        # no activation at final layer
        out_h3 = tf.matmul(out_h2, W) + b
        
    return out_h1, out_h2, out_h3

## Loss Graph
def loss(batch_logits, batch_ref):
    # batch_logits : inference result [batch_size, output_dim] ex) mnist case : output_dim =10
    # batch_ref    : reference        [batch_size]
    
    
    # calculate loss
    # from : https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#sparse_softmax_cross_entropy_with_logits
    
    _batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(batch_logits, batch_ref)
    
    # return : scala_value
    # loss collapsed! (batch-wise)
    _loss = tf.reduce_mean(_batch_loss)
    return _loss
   
def updator(loss_op):
    # do update for minimizing loss from loss_op
    #
    
    opt = tf.train.GradientDescentOptimizer
    #opt = tf.train.AdamOptimizer
    
    optimizer = opt(learning_rate=0.1)
    update_op = optimizer.minimize(loss_op)
    
    return update_op



## Graph Connection && Placeholder
# placeholder for batch input & output
pl_batch_input = tf.placeholder(tf.float32, shape=(batch_size, 784), name='pl_image')
pl_batch_ref   = tf.placeholder(tf.int32,   shape=(batch_size),      name='pl_label')

# flow : inference --> loss --> updator --> training loop
_, _, batch_logits = inference(pl_batch_input)    # batch_input <-- will be placed
loss_op = loss(batch_logits, pl_batch_ref)        # batch_ref <--- will be placed 
updator_op = updator(loss_op)

# ## Initalize Session
sess = tf.Session()
# init all variable before training loop
init_op = tf.initialize_all_variables()
sess.run(init_op)

# prepare timeline
run_metadata = tf.RunMetadata()  # <-- prepare collecting metadata
trace_file = open('timeline.ctf.json', 'w') # <-- file to store

# ## Training Loop

epoch = 0
step = 0 
start_index = 0
print "batch_size : ", batch_size
epoch_losses = []
while True:
    if epoch > 100: break
    step += 1
    
    ## --- every step ----
    ## do params. update batchwise
    batch_input, batch_refs = batch_reader(start_index)
    if len(batch_input) < 1 : 
        # epoch is finished
        epoch += 1
        start_index = 0
        print "\t >> Epoch Step {} loss {}".format(epoch, np.mean(epoch_losses))
        epoch_losses = []
        continue 
    
    feed_dict = {
                    pl_batch_input : batch_input,
                    pl_batch_ref   : batch_refs,
                }
    loss_value, _ = sess.run( [loss_op, updator_op], 
                             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), # <-- timeline!
                             run_metadata=run_metadata,
                             feed_dict=feed_dict
                            )
    """if step % 10 == 0: 
        print "Epoch : {} Step : {} Loss : {}".format(epoch, step, loss_value)
    """
    epoch_losses.append( loss_value ) 
    start_index += batch_size 

trace = timeline.Timeline(step_stats=run_metadata.step_stats)
trace_file.write(trace.generate_chrome_trace_format())


trace_file.close()
