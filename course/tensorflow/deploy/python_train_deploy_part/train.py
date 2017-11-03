'''
Training MNIST code 
Author : Sangkeun Jung (2017)
'''

from __future__ import print_function
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import math 


import keras
from keras.datasets import mnist  # keras is only used for dataset loading
import pickle as pkl 

from common import prepare_data

class MnistModel:
    def __init__(self, batch_size=100):
        num_class = 10 
        self.batch_size = batch_size
        self.num_class  = 10
        self.input_dim  = 28 * 28


        self._input()
        self._forward()
        self._loss(self.logits, self.pl_labels, self.num_class)
        self._train()

        self.show_model_params()


    def _input(self):
        self.pl_images = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_dim), name='pl_image')
        self.pl_labels = tf.placeholder(tf.int32,   shape=(self.batch_size),                 name='pl_label')



    def _forward(self):
        # Build neural network model 

        hidden_1 = 128
        hidden_2 = 32
        input_dim = 28 * 28 

        def variable_init_2D(num_input, num_output):
            """Initialize weight matrix using truncated normal method
                check detail from Lecun (98) paper. 
                - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
            """
            init_tensor = tf.truncated_normal([num_input, num_output], stddev=1.0 / math.sqrt(float(num_input)))
            return init_tensor

        # X --> Hidden 1
        with tf.name_scope('hidden1') as scope:
            W      = tf.Variable( variable_init_2D(input_dim, hidden_1), name='W')
            b      = tf.Variable( tf.zeros([hidden_1]), name='b')
            out_h1 = tf.nn.relu(  tf.matmul(self.pl_images, W) + b)
  
        # Hidden 1 --> Hidden 2
        with tf.name_scope('hidden2') as scope:
            W      = tf.Variable( variable_init_2D(hidden_1, hidden_2), name='W')
            b      = tf.Variable( tf.zeros([hidden_2]), name='b')
            out_h2 = tf.nn.relu(  tf.matmul(out_h1, W) + b)

        # Hidden 2 --> Y
        with tf.name_scope('softmax') as scope:
            W      = tf.Variable( variable_init_2D(hidden_2, self.num_class), name='W')
            b      = tf.Variable(tf.zeros([self.num_class]), name='b')
            logits = tf.matmul(out_h2, W) + b

        self.logits = logits

    def _loss(self, logits, labels, num_class):
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
        concated      = tf.concat([indices, labels], 1 )
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, num_class]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=onehot_labels,
                                                                name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        print("Loss : ", loss)
        self.loss = loss

    def _train(self):
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.train_op = optimizer.minimize(self.loss)


    def show_model_params(self):
      print("-----------------------------")
      print("Trainable Variables")
      print("-----------------------------")
      for v in tf.trainable_variables(): print(v.name)
      print("-----------------------------")


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
    

model_dir = "./model_dir"

train_data, test_data = prepare_data("mnist.data.pkl")

batch_size = 100 
model = MnistModel(batch_size)    
saver = tf.train.Saver()

sv = tf.train.Supervisor(logdir=model_dir, saver=saver)

with sv.managed_session(config=tf.ConfigProto()) as sess:

    images, labels = train_data
    epoch = 0 
    while not sv.should_stop():
        fetches = [model.loss, model.train_op]

        for batch_images, batch_labels in next_batch(images, labels, batch_size):
            feed_dict = { 
                model.pl_images: batch_images,
                model.pl_labels: batch_labels,
            }

            _loss, _ = sess.run(fetches, feed_dict)
            print("Epoch {:03d}\tLoss={:5.3f}".format(epoch, _loss))

        if epoch == 100: break 

        epoch += 1


    # graph export
    meta_graph_def = tf.train.export_meta_graph(filename=os.path.join(model_dir, 'model.meta'))

    # save final parameters
    saver.save(sess, os.path.join(model_dir, 'final_model'))

