# -- python 2/3 compatibility --- #
from __future__ import print_function
from __future__ import unicode_literals   # at top of module
from __future__ import absolute_import    # To make Py2 code safer (more like Py3) by preventing implicit relative imports, you can also add this to the top:
# ------------------------------- #
import os, sys
import time
import tensorflow as tf
import numpy as np 
from sentiment_model import Sentiment
import codecs 

def run_train(dataset, hps, logdir):
    with tf.variable_scope("model"):
        model = Sentiment(hps, "train")

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=logdir,
                             summary_op=None,  # Automatic summaries don't work with placeholders.
                             global_step=model.global_step,
                             save_summaries_secs=30,
                             save_model_secs=120 * 5)

    tf_config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=2,
                            inter_op_parallelism_threads=20)

    with sv.managed_session(config=tf_config) as sess:
        local_step       = 0
        prev_global_step = sess.run(model.global_step)
        prev_time        = time.time()
        data_iterator    = dataset.iterate_forever(hps.batch_size, hps.num_steps)

        while not sv.should_stop():
            fetches = [model.global_step, model.loss, model.train_op]
            x, y, w = next(data_iterator)
            fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w})

            local_step += 1

            if local_step < 10 or local_step % 10 == 0:
                cur_time  = time.time()
                
                _global_step = fetched[0]
                _loss        = fetched[1]
                print("Step %d, time = %.2fs, train loss = %.4f" % (_global_step, cur_time - prev_time, _loss))
                prev_time = cur_time
    sv.stop()

# from https://github.com/rafaljozefowicz/lm
def load_from_checkpoint(saver, logdir):
    sess = tf.get_default_session()
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Restores from checkpoint with relative path.
            #print( ckpt.model_checkpoint_path )
            #saver.restore(sess, os.path.join(logdir, ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    return False

# from https://github.com/rafaljozefowicz/lm
class CheckpointLoader(object):
    def __init__(self, saver, global_step, logdir):
        self.saver              = saver
        self.global_step_tensor = global_step
        self.logdir             = logdir
        # TODO(rafal): make it restart-proof?
        self.last_global_step = 0

    def load_checkpoint(self):
        while True:
            if load_from_checkpoint(self.saver, self.logdir):
                global_step = int(self.global_step_tensor.eval())
                if global_step <= self.last_global_step:
                    print("Waiting for a new checkpoint...")
                    time.sleep(60)
                    continue
                print("Succesfully loaded model at step=%s." % global_step)
            else:
                print("No checkpoint file found. Waiting...")
                time.sleep(60)
                continue
            self.last_global_step = global_step
            return True

def run_eval(dataset, hps, logdir):
    with tf.variable_scope("model"):
        hps.keep_prob = 1.0
        model = Sentiment(hps, "eval")

    saver = tf.train.Saver()

    # Use only 4 threads for the evaluation.
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=20,
                            inter_op_parallelism_threads=1)

    sess = tf.Session(config=config)
    sw = tf.train.SummaryWriter(logdir + "/eval", sess.graph)
    ckpt_loader = CheckpointLoader(saver, model.global_step, logdir + "/train")

    out_f = codecs.open('_out.txt', 'w', encoding='utf-8')
    with sess.as_default():
        if ckpt_loader.load_checkpoint():
            global_step = ckpt_loader.last_global_step
            
            data_iterator = dataset.iterate_once(hps.batch_size, hps.num_steps)
            tf.initialize_local_variables().run()

            total_count = 0
            correct_count = 0 
            for i, (x, y, w) in enumerate(data_iterator):
                out_pred, out_probs = sess.run([model.out_pred, model.out_probs], {model.x: x, model.y: y, model.w: w})

                ## sentence-wise checking
                B = out_pred.shape[0]
                for b in range(B):
                    sentence   = dataset.recover_sentence(x[b]) # reference
                    ref_class  = dataset.recover_class(y[b])
                    pred_class = dataset.recover_class(out_pred[b])

                    _sent = u''.join( [ t for t in sentence if t != dataset._vocab.pad ] ) 
                    if _sent.strip() == '': continue  # no data?

                    diff = 'O' if ref_class == pred_class else 'X'
                    print( u'[{}]\t{}\t{}\t{}\t{}'.format(diff, ref_class, pred_class, out_probs[b][out_pred[b]], _sent ), file=out_f)
                    if diff == 'O': correct_count += 1

                    total_count += 1

            print("Accuracy : {} - {}/{}".format(float(correct_count)/float(total_count), correct_count, total_count))
    out_f.close()



