"""

    Sentiment Analysis 

        Author : Sangkeun Jung (2017)
        - using Tensorflow
"""

import sys, os

# add common to path
from pathlib import Path
common_path = str(Path( os.path.abspath(__file__) ).parent.parent.parent)
sys.path.append( common_path )

from common.nlp.vocab import Vocab
from common.nlp.data_loader import N21TextData
from common.nlp.converter import N21Converter

from dataset import SentimentDataset
from dataset import load_data
from common.ml.hparams import HParams

import numpy as np
import copy 
import time 
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.layers.python.layers import linear

from common.ml.tf.deploy import freeze_graph

print( "Tensorflow Version : ", tf.__version__)

class SentimentAnalysis():
    def __init__(self, hps, mode="train"):
        self.hps = hps
        self.x = tf.placeholder(tf.int32,   [None, hps.num_steps], name="pl_tokens")
        self.y = tf.placeholder(tf.int32,   [None], name="pl_target")
        self.w = tf.placeholder(tf.float32, [None, hps.num_steps], name="pl_weight")
        self.keep_prob = tf.placeholder(tf.float32, [], name="pl_keep_prob")

        ### 4 blocks ###
        # 1) embedding
        # 2) dropout on input embedding
        # 3) sentence encoding using rnn
        # 4) encoding to output classes
        # 5) loss calcaulation

        def _embedding(x):
            # character embedding 
            shape       = [hps.vocab_size, hps.emb_size]
            initializer = tf.initializers.variance_scaling(distribution="uniform", dtype=tf.float32)
            emb_mat     = tf.get_variable("emb", shape, initializer=initializer, dtype=tf.float32)
            input_emb   = tf.nn.embedding_lookup(emb_mat, x)   # [batch_size, sent_len, emb_dim]

            # split input_emb -> num_steps
            step_inputs = tf.unstack(input_emb, axis=1)
            return step_inputs

        def _sequence_dropout(step_inputs, keep_prob):
            # apply dropout to each input
            # input : a list of input tensor which shape is [None, input_dim]
            with tf.name_scope('sequence_dropout') as scope:
                step_outputs = []
                for t, input in enumerate(step_inputs):
                    step_outputs.append( tf.nn.dropout(input, keep_prob) )
            return step_outputs

        def sequence_encoding_n21_rnn(step_inputs, cell_size, scope_name):
            # rnn based N21 encoding (GRU)
            step_inputs = list( reversed( step_inputs ) )
            f_rnn_cell = tf.contrib.rnn.GRUCell(cell_size, reuse=None)
            _inputs = tf.stack(step_inputs, axis=1)
            step_outputs, final_state = tf.contrib.rnn.static_rnn(f_rnn_cell,
                                                step_inputs,
                                                dtype=tf.float32,
                                                scope=scope_name
                                            )
            out = step_outputs[-1]
            return out

        def _to_class(input, num_class):
            out = linear(input, num_class, scope="Rnn2Sentiment") # out = [batch_size, 4]
            return out

        def _loss(out, ref):
            # out : [batch_size, num_class] float - unscaled logits
            # ref : [batch_size] integer
            # calculate loss function using cross-entropy
            batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=ref, name="sentiment_loss") # [batch_size]
            loss = tf.reduce_mean(batch_loss)
            return loss
        
        seq_length    = tf.reduce_sum(self.w, 1) # [batch_size]

        step_inputs   = _embedding(self.x)
        step_inputs   = _sequence_dropout(step_inputs, self.keep_prob)
        sent_encoding = sequence_encoding_n21_rnn(step_inputs, hps.enc_dim, scope_name="encoder")
        out           = _to_class(sent_encoding, hps.num_target_class)
        loss          = _loss(out, self.y) 

        out_probs     = tf.nn.softmax(out, name="out_probs")
        out_pred      = tf.argmax(out_probs, 1, name="out_pred")

        self.loss      = loss
        self.out_probs = out_probs
        self.out_pred  = out_pred

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer, trainable=False)

        if mode == "train":
            optimizer       = tf.train.AdamOptimizer(hps.learning_rate)
            self.train_op   = optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            self.train_op = tf.no_op()


    @staticmethod
    def get_default_hparams():
        return HParams(
            learning_rate     = 0.001,
            keep_prob         = 0.5,
        )




def train(train_id_data, num_vocabs, num_taget_class):
    #
    # train sentiment analysis using given train_id_data
    #
    max_epoch = 100
    model_dir = "./trained_models"
    hps = SentimentAnalysis.get_default_hparams()
    hps.update(
                    batch_size= 100,
                    num_steps = 128,
                    emb_size  = 50,
                    enc_dim   = 100,
                    vocab_size=num_vocabs,
                    num_target_class=num_taget_class
               )

    with tf.variable_scope("model"):
        model = SentimentAnalysis(hps, "train")

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=model_dir,
                             summary_op=None,  
                             global_step=model.global_step)

    # tf assign compatible operators for gpu and cpu 
    tf_config = tf.ConfigProto(allow_soft_placement=True)

    with sv.managed_session(config=tf_config) as sess:
        local_step       = 0
        prev_global_step = sess.run(model.global_step)

        train_data_set = SentimentDataset(train_id_data, hps.batch_size, hps.num_steps)
        losses = []
        while not sv.should_stop():
            fetches = [model.global_step, model.loss, model.train_op]
            a_batch_data = next( train_data_set.iterator )
            y, x, w = a_batch_data
            fetched = sess.run(fetches, {
                                            model.x: x, 
                                            model.y: y, 
                                            model.w: w,

                                            model.keep_prob: hps.keep_prob,
                                        }
                              )

            local_step += 1

            _global_step = fetched[0]
            _loss        = fetched[1]
            losses.append( _loss )
            if local_step < 10 or local_step % 10 == 0:
                epoch = train_data_set.get_epoch_num()
                print("Epoch = {:3d} Step = {:7d} loss = {:5.3f}".format(epoch, _global_step, np.mean(losses)) )
                _loss = []                
                if epoch >= max_epoch : break 

        print("Training is done.")
    sv.stop()

    # model.out_pred, model.out_probs
    freeze_graph(model_dir, "model/out_pred,model/out_probs", "frozen_graph.tf.pb") ## freeze graph with params to probobuf format
    
from tensorflow.core.framework import graph_pb2
def predict(token_vocab, target_vocab, sent):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force to use cpu only (prediction)
    model_dir = "./trained_models"

    # prepare sentence converting
    # to make raw sentence to id data easily
    in_sent = '{}\t{}'.format('___DUMMY_CLASS___', sent)
    pred_data     = N21TextData(in_sent, mode='sentence')
    pred_id_data  = N21Converter.convert(pred_data, target_vocab, token_vocab)
    pred_data_set = SentimentDataset(pred_id_data, 1, 128)

    #
    a_batch_data = next(pred_data_set.predict_iterator) # a result
    b_sentiment_id, b_token_ids, b_weight = a_batch_data

    # Restore graph
    # note that frozen_graph.tf.pb contains graph definition with parameter values in binary format
    _graph_fn =  os.path.join(model_dir, 'frozen_graph.tf.pb')
    with tf.gfile.GFile(_graph_fn, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    with tf.Session(graph=graph) as sess:
        # to check load graph
        #for n in tf.get_default_graph().as_graph_def().node: print(n.name)

        # make interface for input
        pl_token     = graph.get_tensor_by_name('import/model/pl_tokens:0')
        pl_keep_prob = graph.get_tensor_by_name('import/model/pl_keep_prob:0')

        # make interface for output
        out_pred  = graph.get_tensor_by_name('import/model/out_pred:0')
        out_probs = graph.get_tensor_by_name('import/model/out_probs:0')
        

        # predict sentence 
        b_best_pred_index, b_pred_probs = sess.run([out_pred, out_probs], feed_dict={
                                                                                        pl_token : b_token_ids,
                                                                                        pl_keep_prob : 1.0,
                                                                                    }
                                          )

        best_pred_index = b_best_pred_index[0]
        pred_probs = b_pred_probs[0]

        best_target_class = target_vocab.get_symbol(best_pred_index)
        print( pred_probs[best_pred_index] )
        best_prob  = int( pred_probs[best_pred_index] )
        print(best_target_class, best_prob)


if __name__ == '__main__':
    train_id_data, token_vocab, target_vocab = load_data()
    num_vocabs       = token_vocab.get_num_tokens()
    num_target_class = target_vocab.get_num_targets()


    train(train_id_data, num_vocabs, num_target_class)
    predict(token_vocab, target_vocab, '유일한 단점은 방안에 있어도 들리는 소음인데요.')
