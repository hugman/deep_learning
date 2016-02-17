# Deep learning tutorial codes using Tensorflow




# Configuration
To Run the scripts in this tutorial, it's better to set following configurations:
* install tensorflow 0.6 
* ubuntu > 12.04
* python 2.7
* (recommended) GPU cards and Nvidia drivers. 

# Recommended order to check
It's better to investigate source codes in following order 

1. mnist.py  --> basic block of neural network
2. pos_tagger_fcn.py --> how to change input/output parts of mnist.py to handle text (english pos tagging)
3. pos_tagger_fcn_seq.py --> how to change input/output, loss, update of *_fcn.py to handle sentence wise data
4. pos_tagger_rnn_seq.py --> how to change input/output, loss, update of *_fcn_seq.py to use RNN 

2, 3, 4 codes are based on mnist.py 

So some variable names are not changed for better understanding. (ex. var name 'image' instead of 'word' )

# How to use tensorboard to monitor 
For monitoring using tensorbaord:
   You need to launch tensorboard web server as follows:
   
   <code>
	$ ./nohup tensorboard --logdir /local/log-tensorboard &
   </code>
   
   in here, log-tensorboard can be changed by users.

