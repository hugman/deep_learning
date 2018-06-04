"""

    Korean Sentiment Analysis

        Author : Sangkeun Jung (2017)
        - using PyTorch
"""

import sys, os

# add common to path
from pathlib import Path
common_path = str(Path( os.path.abspath(__file__) ).parent.parent.parent)
sys.path.append( common_path )

from common.nlp.vocab import Vocab
from common.nlp.data_loader import N21TextData
from common.nlp.converter import N21Converter
from common.nlp.dataset import Dataset


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.autograd import Variable
import numpy as np

import copy 
torch.manual_seed(1) # to reproduce same result 
use_gpu = False 

print( "PyTorch Version : ", torch.__version__)

# for sentiment dataset
class SentimentDataset(Dataset):
    def _iterate(self, index_gen):
        B = self.batch_size
        N = self.num_steps

        # vectorize id data
        sentiment  = np.zeros([B],    np.int64)  
        token      = np.zeros([B, N], np.int64)
        weight     = np.zeros([B, N], np.int64)

        while True:
            sentiment[:]  = 0
            token[:]      = 0
            weight[:]     = 0

            for b in range(B):
                try:
                    while True:
                        index = next(index_gen)
                        _num_steps = len( self.data[index].token_ids )
                        if _num_steps <= N: break 

                    _sentiment_id = copy.deepcopy( self.data[index].target_id )
                    _token_ids    = copy.deepcopy( self.data[index].token_ids )

                    # fill pad for weight
                    _weight_ids   = [0] * self.num_steps
                    for _idx, _ in enumerate(_token_ids): _weight_ids[_idx] = 1

                    # fill pad to token
                    _token_ids += [1] * ( self.num_steps - len( _token_ids ) ) 

                    # output
                    sentiment[b] = -1 if _sentiment_id is None else _sentiment_id

                    # input
                    # for padding, make it reverse
                    _token_ids.reverse()
                    token[b]  = _token_ids
                    weight[b] = _weight_ids

                except StopIteration:
                    pass
            if not np.any(weight):
                return
            yield sentiment, token, weight  # tuple for (target, input)



class SentimentAnalysis(nn.Module):
    def __init__(self, num_vocabs, num_target_class, batch_size, num_steps, use_gpu=False):
        super(SentimentAnalysis, self).__init__()
        self.embedding_dim = 50
        self.encoder_dim   = 30
        self.batch_size    = batch_size
        self.num_steps     = num_steps
        self.num_rnn_layers = 1
        self.use_gpu       = use_gpu

        self.word_embeddings = nn.Embedding(num_vocabs, self.embedding_dim)
        self.lstm            = nn.LSTM(self.embedding_dim, self.encoder_dim) # [batch_size, 50] --> [batch_size, 30]        
        self.to_class        = nn.Linear(self.encoder_dim, num_target_class)  # [batch_size, 30] --> [batch_size, num_target_class]    
        
        self.initial_state = self.reset_initial_state() 
        # self.reset_initial_state() is importatnt 
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.

    
    def reset_initial_state(self):
        # in LSTM spec.
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_rnn_layers, self.batch_size, self.encoder_dim).cuda())
            c0 = Variable(torch.zeros(self.num_rnn_layers, self.batch_size, self.encoder_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_rnn_layers, self.batch_size, self.encoder_dim))
            c0 = Variable(torch.zeros(self.num_rnn_layers, self.batch_size, self.encoder_dim))
        return (h0, c0)

    def forward(self, b_token_ids):
        embedding = self.word_embeddings(b_token_ids) # [batch_size, num_steps, emb_dim]

        # to feed LSTM, we need to reshape the embedding results 
        # lstm() accepts input shape : [num_steps, batch_size, input_dim ]
        _input = torch.transpose(embedding, 0, 1)
        lstm_out, self.initial_state = self.lstm(_input, self.initial_state)

        # lstm_out : [num_steps, batch_size, output_dim]  
        last_out = lstm_out[-1] # final rnn output keeps all sequence information
        logit    = self.to_class(last_out) # [batch_size, num_target_class]
        return logit

def load_data():

    # vocab loader
    token_vocab_fn  = os.path.join( os.path.dirname(__file__), 'data', 'token.vocab.txt')
    token_vocab     = Vocab(token_vocab_fn, mode='token')
    target_vocab_fn = os.path.join( os.path.dirname(__file__), 'data', 'target.vocab.txt')
    target_vocab    = Vocab(target_vocab_fn, mode='target')


    # load train data 
    train_data_fn  = os.path.join( os.path.dirname(__file__), 'data', 'train.sent_data.txt')
    #train_data_fn  = os.path.join( os.path.dirname(__file__), 'data', 'small.train.sent_data.txt')
    train_txt_data = N21TextData(train_data_fn)

    # convert text data to id data
    train_id_data  = N21Converter.convert(train_txt_data, target_vocab, token_vocab)
    
    return train_id_data, token_vocab, target_vocab

def train(train_id_data, num_vocabs, num_taget_class):
    #
    # train sentiment analysis using given train_id_data
    #
    max_epoch  = 100
    batch_size = 100
    num_steps  = 128

    model         = SentimentAnalysis( num_vocabs, num_taget_class, batch_size, num_steps, use_gpu=True )
    if use_gpu: model = model.cuda()
    optimizer     = optim.Adagrad(model.parameters(), lr=0.01)

    train_data_set = SentimentDataset(train_id_data, batch_size, num_steps)
    step = 0 
    losses = []
    while True: # for each batch
        
        a_batch_data = next( train_data_set.iterator )
        b_sentiment_id, b_token_ids, b_weight = a_batch_data

        # convert numpy to pytorch
        b_sentiment_id = Variable( torch.from_numpy(b_sentiment_id) )
        b_token_ids    = Variable( torch.from_numpy(b_token_ids) )

        # clear gradients for new data instance
        optimizer.zero_grad()

        # init initial states for LSTM layer
        model.initial_state = model.reset_initial_state()

        # feed-forward (inference)
        logit = model(b_token_ids)
        loss = F.cross_entropy(logit, b_sentiment_id)
            
        loss.backward()
        optimizer.step()

        loss_value = float( loss.data.numpy() )
        losses.append(loss_value)

        if step % 100 == 0:
            epoch = train_data_set.get_epoch_num()
            print("Epoch = {:3d} Iter = {:7d} loss = {:5.3f}".format(epoch, step, np.mean(losses)) )
            losses = []
            if epoch >= max_epoch : break 
        step += 1

    # save model 
    model_fn = os.path.join( os.path.dirname(__file__), 'model.pt') 
    torch.save(model, model_fn)
    print("Model saved at {}".format(model_fn) )


def predict(token_vocab, target_vocab, sent):
    # load trained model 
    model_fn = os.path.join( os.path.dirname(__file__), 'model.pt') 
    model = torch.load(model_fn)

    batch_size = 1
    num_steps  = model.num_steps

    in_sent = '{}\t{}'.format('___DUMMY_CLASS___', sent)

    # to make raw sentence to id data easily
    pred_data     = N21TextData(in_sent, mode='sentence')
    pred_id_data  = N21Converter.convert(pred_data, target_vocab, token_vocab)
    pred_data_set = SentimentDataset(pred_id_data, batch_size, num_steps)

    #
    a_batch_data = next(pred_data_set.predict_iterator) # a result
    b_sentiment_id, b_token_ids, b_weight = a_batch_data

    # convert numpy to pytorch
    b_sentiment_id = Variable( torch.from_numpy(b_sentiment_id) )
    b_token_ids    = Variable( torch.from_numpy(b_token_ids) )

    # feed-forward (inference)
    model.batch_size = batch_size 
    model.initial_state = model.reset_initial_state()
    logit = model(b_token_ids)
    np_probs   = F.softmax(logit).data[0].numpy() # to get probability

    # torch to numpy 
    best_index = np_probs.argmax()
    best_prob  = np_probs[best_index]
    best_target_class = target_vocab.get_symbol(best_index)

    print("[Prediction] {} ===> {}".format(sent, best_target_class))


if __name__ == '__main__':
    train_id_data, token_vocab, target_vocab = load_data()
    num_vocabs       = token_vocab.get_num_tokens()
    num_target_class = target_vocab.get_num_targets()

    train(train_id_data, num_vocabs, num_target_class)

    print()

    predict(token_vocab, target_vocab, '유일한 단점은 방안에 있어도 들리는 소음인데요.')
