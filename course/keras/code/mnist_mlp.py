'''
Keras DNN script for learning DNN concepts

Author : Sangkeun Jung (2017)

This code is based on https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

### Code Building Blocks ###
# 1) Data preparation (train data : test data)
#       --> download data from web
#       --> preprocessing
# 2) DNN Model Design
# 3) Iterate train block

batch_size = 128
num_classes = 10  # [0, 1, 2, 3, ... 9] 
epochs = 20       # learn training data 20 times


# 1) Data preparation  -------------------------------------------


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
print(x_train.shape[0], 'train samples')
print(x_test.shape[0],  'test samples')

# 1-2) preprocessing (class id 3 to [0,0,0,1,0,0,0,0,0,0])
# convert class vectors to binary class matrices (one-hot distribution)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

# 2) DNN Model Design ---------------------------------------------
model = Sequential()
model.add( Dense(512, activation='relu', input_shape=(784,)) )
model.add( Dropout(0.2) )
model.add( Dense(512, activation='relu') )
model.add( Dropout(0.2) )
model.add( Dense(num_classes, activation='softmax') )

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# 3) iterate train block -------------------------------------------
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:',     score[0])
print('Test accuracy:', score[1])