'''
MNIST Predict based graph only (python version)

Author : Sangkeun Jung (2017)
'''

from __future__ import print_function
from common import prepare_data, load_single_line_data

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf 


model_dir = "./model_dir"


image_1, label_1 = load_single_line_data('ex_1.data')
image_2, label_2 = load_single_line_data('ex_2.data')

# Restore graph

_graph_fn =  os.path.join(model_dir, 'final_model.meta')      # 'AAA'.meta for graph file
saver = tf.train.import_meta_graph( _graph_fn ) 


graph = tf.get_default_graph()

with tf.Session(config=tf.ConfigProto()) as sess:
    saver.restore(sess, os.path.join(model_dir, 'final_model'))   # just use 'AAA' to restore parameters
    print(graph)

    # placeholder ==> operation
    pl_image = graph.get_operation_by_name('pl_image')
    print(pl_image)
    
    # tensor ==> tensor
    
    

