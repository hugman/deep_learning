'''
Common data code
Author : Sangkeun Jung (2017)
'''

from __future__ import print_function
import os 

import tensorflow as tf
import math 

import keras
from keras.datasets import mnist  # keras is only used for dataset loading
import pickle as pkl 

def prepare_data(fn):
    # Data preparation  -------------------------------------------

    # if the *.pkl file exists? skip 
    if os.path.exists(fn) and os.stat(fn).st_size != 0:
        print("Data is already prepared with pickle file - {}   .. Skipped".format(fn))

        with open(fn, 'rb') as f:
            train_data, test_data = pkl.load(f)
        return train_data, test_data

    # 1-1) down load data 
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train : [60000, 28, 28]
    # y_train : [60000, ]
    # x_test  : [10000, 28, 28]
    # y_test  : [10000, ]

    # 1-2) preprocessing (2D --> 1D)
    x_train = x_train.reshape(60000, 784)  # [60000, 28, 28] --> [60000, 28*28]
    x_test  = x_test.reshape(10000, 784)   # [10000, 28, 28] --> [10000, 28*28]
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    # 1-2) preprocessing (normalize to 0~1.0)
    x_train /= 255
    x_test  /= 255

    # dump as pickle 
    train_data = ( x_train, y_train )
    test_data  = ( x_test, y_test )
    _data      = ( train_data, test_data)

    with open(fn, 'wb') as f:
        pkl.dump(_data, f)

    return train_data, test_data


def load_single_line_data(fn):
    line = open(fn).readlines()[0]
    line = line.rstrip()

    label, num_part = line.split('\t')

    label = int(label)
    image = [float(x) for x in num_part.split(',')]

    return image, label

    



