#!/bin/bash

echo Start tensorboard to monitor training and evaluation
nohup tensorboard --port=39790 --logdir=./log_dir/train/ &

