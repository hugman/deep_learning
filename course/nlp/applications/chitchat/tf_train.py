"""
    Chitchat dialog modeling using Sequence to Sequence framework

        Author : Sangkeun Jung (2017)
        - using Tensorflow
"""

import sys, os

# add common to path
from pathlib import Path
common_path = str(Path( os.path.abspath(__file__) ).parent.parent.parent)
sys.path.append( common_path )

from common.nlp.vocab import Vocab
from common.nlp.data_loader import N2MTextData
from common.nlp.converter import N2MConverter

from dataset import ChitchatDataset
from dataset import load_data
from common.ml.hparams import HParams


import numpy as np
import copy 
import time 
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.layers.python.layers import linear
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.seq2seq import sequence_loss

from common.ml.tf.deploy import freeze_graph


print( "Tensorflow Version : ", tf.__version__)

class Chitchat():
    def __init__(self, hps, mode="train"):
        self.hps = hps
        self.x = tf.placeholder(tf.int32,    [None, hps.src_num_steps], name="pl_src_tokens")
        self.y = tf.placeholder(tf.int32,    [None, hps.tar_num_steps], name="pl_tar_tokens")
        
        self.sw = tf.placeholder(tf.float32, [None, hps.src_num_steps], name="pl_src_weight")
        self.tw = tf.placeholder(tf.float32, [None, hps.tar_num_steps], name="pl_tar_weight")

        self.keep_prob = tf.placeholder(tf.float32, [], name="pl_keep_prob")

        ### 4 blocks ###
        # 1) embedding
        # 2) dropout on input embedding
        # 3) sentence source text encoding using rnn
        # 4) decoding target text using another rnn
        # 5) loss calcaulation

        def _embedding(x):
            # character embedding 
            shape       = [hps.src_vocab_size, hps.emb_size]
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

        def sequence_decoding_n2n(step_inputs, seq_length, cell_size, scope_name):
            # rnn based N2N encoding and output
            f_rnn_cell = tf.contrib.rnn.GRUCell(cell_size, reuse=False)
            _inputs    = tf.stack(step_inputs, axis=1)
            step_outputs, final_state = tf.contrib.rnn.static_rnn(f_rnn_cell,
                                                step_inputs,
                                                dtype=tf.float32,
                                                scope=scope_name
                                            )
            return step_outputs

        def _to_class_n2n(step_inputs, num_class):
            T = len(step_inputs)
            step_output_logits = []
            for t in range(T):
                # encoder to linear(map)
                out = step_inputs[t]
                if t==0: out = linear(out, num_class, scope="Rnn2Target")
                else:    out = linear(out, num_class, scope="Rnn2Target", reuse=True)
                step_output_logits.append(out)
            return step_output_logits

        def _loss(step_outputs, step_refs, weights):
            # step_outputs : a list of [batch_size, num_class] float32 - unscaled logits
            # step_refs    : [batch_size, num_steps] int32
            # weights      : [batch_size, num_steps] float32
            # calculate sequence wise loss function using cross-entropy
            _batch_output_logits = tf.stack(step_outputs, axis=1)
            loss = sequence_loss(
                                    logits=_batch_output_logits,        
                                    targets=step_refs,
                                    weights=weights
                                )
            return loss
        
        src_seq_length    = tf.reduce_sum(self.sw, 1) # [batch_size]
        tar_seq_length    = tf.reduce_sum(self.tw, 1) # [batch_size]

        # encoding source sentence
        enc_step_inputs   = _embedding(self.x)
        enc_step_inputs   = _sequence_dropout(enc_step_inputs, self.keep_prob)
        sent_encoding     = sequence_encoding_n21_rnn(enc_step_inputs, hps.enc_dim, scope_name="encoder")

        # decoding 
        dec_step_inputs   = [sent_encoding] * hps.tar_num_steps
        step_outputs      = sequence_decoding_n2n(dec_step_inputs, tar_seq_length, hps.dec_dim, scope_name="decoder")

        # to target text
        step_outputs      = _to_class_n2n(step_outputs, hps.tar_vocab_size)

        self.loss = _loss(step_outputs, self.y, self.tw)

        # step_preds and step_out_probs
        step_out_probs = []
        step_out_preds = []
        for _output in step_outputs:
            _out_probs  = tf.nn.softmax(_output)
            _out_pred   = tf.argmax(_out_probs, 1)

            step_out_probs.append(_out_probs)
            step_out_preds.append(_out_pred)

        # stack for interface
        self.step_out_probs = tf.stack(step_out_probs, axis=1, name="step_out_probs")
        self.step_out_preds = tf.stack(step_out_preds, axis=1, name="step_out_preds")

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer, trainable=False)

        if mode == "train":
            optimizer       = tf.train.AdamOptimizer(hps.learning_rate)
            self.train_op   = optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            self.train_op = tf.no_op()

        for v in tf.trainable_variables(): print(v.name)

    @staticmethod
    def get_default_hparams():
        return HParams(
            learning_rate     = 0.001,
            keep_prob         = 0.5,
        )


def train(train_id_data, src_num_vocabs, tar_num_vocabs):
    #
    # train sentiment analysis using given train_id_data
    #
    max_epoch = 300
    model_dir = "./trained_models"
    hps = Chitchat.get_default_hparams()
    hps.update(
                    batch_size= 100,

                    src_num_steps = 128,
                    tar_num_steps = 128,

                    emb_size  = 50,
                    
                    enc_dim   = 100,
                    dec_dim   = 100,

                    src_vocab_size=src_num_vocabs,
                    tar_vocab_size=tar_num_vocabs,
               )

    with tf.variable_scope("model"):
        model = Chitchat(hps, "train")

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=model_dir,
                             summary_op=None,  
                             global_step=model.global_step)

    # tf assign compatible operators for gpu and cpu 
    tf_config = tf.ConfigProto(allow_soft_placement=True)

    with sv.managed_session(config=tf_config) as sess:
        local_step       = 0
        prev_global_step = sess.run(model.global_step)

        train_data_set = ChitchatDataset(train_id_data, hps.batch_size, hps.src_num_steps, hps.tar_num_steps)
        losses = []
        while not sv.should_stop():
            fetches = [model.global_step, model.loss, model.train_op]
            a_batch_data = next( train_data_set.iterator )

            x, y, sw, tw = a_batch_data


            fetched = sess.run(fetches, {
                                            model.x:  x, 
                                            model.y:  y, 
                                            model.sw: sw,
                                            model.tw: tw,

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
    freeze_graph(model_dir, "model/step_out_preds,model/step_out_probs", "frozen_graph.tf.pb") ## freeze graph with params to probobuf format
    
from tensorflow.core.framework import graph_pb2
def predict(src_token_vocab, tar_token_vocab, sent):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force to use cpu only (prediction)
    model_dir = "./trained_models"

    # prepare sentence converting
    # to make raw sentence to id data easily
    pred_data     = N2MTextData(sent, mode='sentence')
    pred_id_data  = N2MConverter.convert(pred_data, src_token_vocab, tar_token_vocab)
    pred_data_set = ChitchatDataset(pred_id_data, 1, 128, 128)
    #
    a_batch_data = next(pred_data_set.predict_iterator) # a result
    b_src_token_ids, b_tar_token_ids, b_src_weight, b_tar_weight = a_batch_data

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
        pl_src_token     = graph.get_tensor_by_name('import/model/pl_src_tokens:0')
        pl_src_weight    = graph.get_tensor_by_name('import/model/pl_src_weight:0')
        pl_keep_prob     = graph.get_tensor_by_name('import/model/pl_keep_prob:0')

        # make interface for output
        step_out_preds = graph.get_tensor_by_name('import/model/step_out_preds:0')
        step_out_probs = graph.get_tensor_by_name('import/model/step_out_probs:0')
        

        # predict sentence 
        b_best_step_pred_indexs, b_step_pred_probs = sess.run([step_out_preds, step_out_probs], 
                                                              feed_dict={
                                                                            pl_src_token  : b_src_token_ids,
                                                                            pl_tar_weight : b_tar_weight,
                                                                            pl_keep_prob : 1.0,
                                                                        }
                                                             )
        best_step_pred_indexs = b_best_step_pred_indexs[0]
        step_pred_probs       = b_step_pred_probs[0]

        step_best_targets      = []
        step_best_target_probs = []
        for time_step, best_pred_index in enumerate(best_step_pred_indexs):
            _target_symbol = target_vocab.get_symbol(best_pred_index)
            step_best_targets.append( _target_symbol )
            _prob = step_pred_probs[time_step][best_pred_index]
            step_best_target_probs.append( _prob ) 

        print("Input  : ", sent)
        print("Output : ", step_best_targets)



if __name__ == '__main__':
    train_id_data, src_token_vocab, tar_token_vocab = load_data()
    src_num_vocabs = src_token_vocab.get_num_tokens()
    tar_num_vocabs = tar_token_vocab.get_num_tokens()

    train_data_set = ChitchatDataset(train_id_data, 100, 128, 128)

    train(train_id_data, src_num_vocabs, tar_num_vocabs)
    
    #predict(token_vocab, target_vocab, '의정지기단은 첫 사업으로 45 명 시의원들의 선거 공약을 수집해 개인별로 카드를 만들었다.')
    #predict(token_vocab, target_vocab, '한국소비자보호원은 19일 시판중인 선물세트의 상당수가 과대 포장된 것으로 드러났다고 밝혔다.')
