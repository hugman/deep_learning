#-*- coding: utf-8 -*-
"""
    Sentiment Analysis NN Model

    Author : Sangkeun Jung (hugmanskj@gmail.com, 2017)
"""
import tensorflow as tf
from hparams import HParams

from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib.layers.python.layers import linear

class Sentiment(object):
    def __init__(self, hps, mode="train"):
        self.hps = hps
        self.x = tf.placeholder(tf.int32, [hps.batch_size, hps.num_steps])
        self.y = tf.placeholder(tf.int32, [hps.batch_size])
        self.w = tf.placeholder(tf.int32, [hps.batch_size, hps.num_steps])

        ### 4 blocks ###
        # 1) embedding
        # 2) dropout on input embedding
        # 3) sentence encoding using rnn
        # 4) encoding to output classes
        # 5) loss calcaulation

        def _embedding(x):
            # character embedding 
            shape       = [hps.vocab_size, hps.emb_size]
            initializer = tf.uniform_unit_scaling_initializer(dtype=tf.float32, full_shape=shape)
            emb_mat     = tf.get_variable("emb", shape, initializer=initializer, dtype=tf.float32)

            input_emb   = tf.nn.embedding_lookup(emb_mat, x)   # [batch_size, sent_len, emb_dim]

            # split input_emb -> num_steps
            step_inputs = tf.unpack(input_emb, axis=1)
            return step_inputs

        def _sequence_dropout(step_inputs, keep_prob):
            # apply dropout to each input
            # input : a list of input tensor which shape is [None, input_dim]
            with tf.name_scope('sequence_dropout') as scope:
                step_outputs = []
                for t, input in enumerate(step_inputs):
                    step_outputs.append( tf.nn.dropout(input, keep_prob) )
            return step_outputs

        def _sentence_encoding(step_inputs, seq_length, cell_size):
            f_rnn_cell = tf.nn.rnn_cell.LSTMCell(cell_size)
            b_rnn_cell = tf.nn.rnn_cell.LSTMCell(cell_size)
            step_rnn_logits, step_f_rnn_state, step_b_rnn_state = rnn.bidirectional_rnn(f_rnn_cell, 
                                                                                        b_rnn_cell, 
                                                                                        step_inputs, 
                                                                                        dtype=tf.float32, 
                                                                                        scope='birnn')

            sent_encoding = step_rnn_logits[0]

            """
            f_rnn_cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
            b_rnn_cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
            _inputs = tf.pack(step_inputs, axis=1)
            outputs, states, = rnn.bidirectional_dynamic_rnn(f_rnn_cell,
                                                             b_rnn_cell,
                                                             _inputs,
                                                             sequence_length=tf.cast(seq_length, tf.int64),
                                                             time_major=False,
                                                             dtype=tf.float32, 
                                                             scope='birnn'
                                                            )

            output_fw, output_bw = outputs
            states_fw, states_bw = states

            _concated = tf.concat(1, [output_fw, output_bw])

            step_concated = tf.unpack(_concated, axis=1)
            #sent_encoding = tf.concat(1, [step_concated[0], step_concated[-1]]) # [batch_size, cell_size*2]
            sent_encoding = step_concated[0] # [batch_size]
            print(sent_encoding)
            """

            return sent_encoding

        def _to_class(input, num_class):
            out = linear(input, num_class, scope="Rnn2Sentiment") # out = [batch_size, 4]
            return out

        def _loss(out, ref):
            # out : [batch_size, num_class] float - unscaled logits
            # ref : [batch_size] integer
            # calculate loss function using cross-entropy
            batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(out, ref, name="sentiment_loss") # [batch_size]
            loss = tf.reduce_mean(batch_loss)
            return loss
        
        seq_length    = tf.reduce_sum(self.w, 1) # [batch_size]

        step_inputs   = _embedding(self.x)
        step_inputs   = _sequence_dropout(step_inputs, hps.keep_prob)
        sent_encoding = _sentence_encoding(step_inputs, seq_length, 300)
        out           = _to_class(sent_encoding, 4)
        loss          = _loss(out, self.y) 

        out_probs     = tf.nn.softmax(out)
        out_pred      = tf.argmax(out_probs, 1)

        self.loss      = loss
        self.out_probs = out_probs
        self.out_pred  = out_pred

        tf.scalar_summary("model/loss", self.loss)
        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer, trainable=False)

        if mode == "train":
            #optimizer       = tf.train.AdagradOptimizer(hps.learning_rate, initial_accumulator_value=1.0)
            optimizer       = tf.train.AdamOptimizer(hps.learning_rate)
            self.train_op   = optimizer.minimize(self.loss, global_step=self.global_step)
            self.summary_op = tf.merge_all_summaries()
        else:
            self.train_op = tf.no_op()


    @staticmethod
    def get_default_hparams():
        return HParams(
            batch_size        = 100,
            num_steps         = 128,
            
            learning_rate     = 0.01,
            num_delayed_steps = 150,
            keep_prob         = 0.9,

            vocab_size        = 1562,
            emb_size          = 50,
        )
