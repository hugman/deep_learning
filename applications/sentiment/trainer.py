#-*- coding: utf-8 -*-
"""
    Sentiment Analysis Tensorflor Example

    - Trainer / Predictor

    Author : Sangkeun Jung (hugmanskj@gmail.com, 2017)
"""
# -- python 2/3 compatibility --- #
from __future__ import print_function
from __future__ import unicode_literals   # at top of module
from __future__ import absolute_import    # To make Py2 code safer (more like Py3) by preventing implicit relative imports, you can also add this to the top:
# ------------------------------- #
import colored_traceback.always

import sys, os
import tensorflow as tf

from io_utils import Vocabulary, Dataset
from run_utils import run_train, run_eval
from sentiment_model import Sentiment

flags = tf.flags
flags.DEFINE_string("log_dir"    , "./log_dir", "Logging directory.")
flags.DEFINE_string("data_dir"   , "./data",    "Data directory.") 
flags.DEFINE_string("mode"       , "train",     "Whether to run 'train' or 'eval' model.")
flags.DEFINE_string("hpconfig"   , "",          "Overrides default hyper-parameters.")
FLAGS = flags.FLAGS

def main(_):
    hps   = Sentiment.get_default_hparams().parse(FLAGS.hpconfig)
    vocab = Vocabulary.from_file( os.path.join(FLAGS.data_dir, "sent.vocab.freq.dict"))

    if FLAGS.mode == "train":
        dataset = Dataset(os.path.join(FLAGS.data_dir, "train.sent_data.txt"), vocab)
        run_train(dataset,                  ## dataset
                  hps,                      ## configurations
                  FLAGS.log_dir + "/train") ## loging dir
    elif FLAGS.mode.startswith("eval"):
        dataset = Dataset(os.path.join(FLAGS.data_dir, "test.sent_data.txt"), vocab)
        run_eval(dataset,                  ## dataset
                  hps,                      ## configurations
                  FLAGS.log_dir) ## loging dir


if __name__ == "__main__":
    tf.app.run()