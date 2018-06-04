"""
    Named Entity Recognition 

        Author : Sangkeun Jung (2017)
        - using Tensorflow
"""

import sys, os, inspect

# add common to path
from pathlib import Path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
common_path = str(Path(currentdir).parent.parent)
sys.path.append( common_path )

from common.nlp.vocab import Vocab
from common.nlp.data_loader import N2NTextData
from common.nlp.converter import N2NConverter

from dataset import NERDataset
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

class NER():
    def __init__(self, hps, mode="train"):
        self.hps = hps
        self.x = tf.placeholder(tf.int32,   [None, hps.num_steps], name="pl_tokens")
        self.y = tf.placeholder(tf.int32,   [None, hps.num_steps], name="pl_target")
        self.w = tf.placeholder(tf.float32, [None, hps.num_steps], name="pl_weight")
        self.keep_prob = tf.placeholder(tf.float32, [], name="pl_keep_prob")

        ### 4 blocks ###
        # 1) embedding
        # 2) dropout on input embedding
        # 3) sentence encoding using rnn
        # 4) bidirectional rnn's output to target classes
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

        def sequence_encoding_n2n(step_inputs, seq_length, cell_size):
            # birnn based N2N encoding and output
            f_rnn_cell = tf.contrib.rnn.GRUCell(cell_size, reuse=False)
            b_rnn_cell = tf.contrib.rnn.GRUCell(cell_size, reuse=False)
            _inputs    = tf.stack(step_inputs, axis=1)

            # step_inputs = a list of [batch_size, emb_dim]
            # input = [batch_size, num_step, emb_dim]
            # np.stack( [a,b,c,] )
            outputs, states, = tf.nn.bidirectional_dynamic_rnn( f_rnn_cell,
                                                                b_rnn_cell,
                                                                _inputs,
                                                                sequence_length=tf.cast(seq_length, tf.int64),
                                                                time_major=False,
                                                                dtype=tf.float32,
                                                                scope='birnn',
                                                            )
            output_fw, output_bw = outputs
            states_fw, states_bw = states 

            output       = tf.concat([output_fw, output_bw], 2)
            step_outputs = tf.unstack(output, axis=1)

            final_state  = tf.concat([states_fw, states_bw], 1)
            return step_outputs # a list of [batch_size, enc_dim]

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
        
        seq_length    = tf.reduce_sum(self.w, 1) # [batch_size]

        step_inputs       = _embedding(self.x)
        step_inputs       = _sequence_dropout(step_inputs, self.keep_prob)
        step_enc_outputs  = sequence_encoding_n2n(step_inputs, seq_length, hps.enc_dim)
        step_outputs      = _to_class_n2n(step_enc_outputs, hps.num_target_class)

        self.loss = _loss(step_outputs, self.y, self.w)

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
            learning_rate     = 0.01,
            keep_prob         = 0.5,
        )


def train(train_id_data, num_vocabs, num_taget_class):
    #
    # train sentiment analysis using given train_id_data
    #
    max_epoch = 300
    model_dir = "./trained_models"
    hps = NER.get_default_hparams()
    hps.update(
                    batch_size= 100,
                    num_steps = 128,
                    emb_size  = 50,
                    enc_dim   = 100,
                    vocab_size=num_vocabs,
                    num_target_class=num_taget_class
               )

    with tf.variable_scope("model"):
        model = NER(hps, "train")

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=model_dir,
                             summary_op=None,  
                             global_step=model.global_step)

    # tf assign compatible operators for gpu and cpu 
    tf_config = tf.ConfigProto(allow_soft_placement=True)

    with sv.managed_session(config=tf_config) as sess:
        local_step       = 0
        prev_global_step = sess.run(model.global_step)

        train_data_set = NERDataset(train_id_data, hps.batch_size, hps.num_steps)
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
    freeze_graph(model_dir, "model/step_out_preds,model/step_out_probs", "frozen_graph.tf.pb") ## freeze graph with params to probobuf format
    
from tensorflow.core.framework import graph_pb2
def predict(token_vocab, target_vocab, sent):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force to use cpu only (prediction)
    model_dir = "./trained_models"

    # prepare sentence converting
    # to make raw sentence to id data easily
    pred_data     = N2NTextData(sent, mode='sentence')
    pred_id_data  = N2NConverter.convert(pred_data, target_vocab, token_vocab)
    pred_data_set = NERDataset(pred_id_data, 1, 128)
    #
    a_batch_data = next(pred_data_set.predict_iterator) # a result
    b_nes_id, b_token_ids, b_weight = a_batch_data

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
        pl_weight    = graph.get_tensor_by_name('import/model/pl_weight:0')
        pl_keep_prob = graph.get_tensor_by_name('import/model/pl_keep_prob:0')

        # make interface for output
        step_out_preds = graph.get_tensor_by_name('import/model/step_out_preds:0')
        step_out_probs = graph.get_tensor_by_name('import/model/step_out_probs:0')
        

        # predict sentence 
        b_best_step_pred_indexs, b_step_pred_probs = sess.run([step_out_preds, step_out_probs], 
                                                              feed_dict={
                                                                            pl_token  : b_token_ids,
                                                                            pl_weight : b_weight,
                                                                            pl_keep_prob : 1.0,
                                                                        }
                                                             )
        best_step_pred_indexs = b_best_step_pred_indexs[0]
        step_pred_probs = b_step_pred_probs[0]

        step_best_targets = []
        step_best_target_probs = []
        for time_step, best_pred_index in enumerate(best_step_pred_indexs):
            _target_class = target_vocab.get_symbol(best_pred_index)
            step_best_targets.append( _target_class )
            _prob = step_pred_probs[time_step][best_pred_index]
            step_best_target_probs.append( _prob ) 

        for idx, char in enumerate(list(sent)):
            print('{}\t{}\t{}'.format(char, step_best_targets[idx], step_best_target_probs[idx]) ) 



if __name__ == '__main__':
    train_id_data, token_vocab, target_vocab = load_data()
    num_vocabs       = token_vocab.get_num_tokens()
    num_target_class = target_vocab.get_num_targets()

    train_data_set = NERDataset(train_id_data, 5, 128)
    train(train_id_data, num_vocabs, num_target_class)
    
    predict(token_vocab, target_vocab, '의정지기단은 첫 사업으로 45 명 시의원들의 선거 공약을 수집해 개인별로 카드를 만들었다.')
    predict(token_vocab, target_vocab, '한국소비자보호원은 19일 시판중인 선물세트의 상당수가 과대 포장된 것으로 드러났다고 밝혔다.')
