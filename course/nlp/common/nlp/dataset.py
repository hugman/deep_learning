
import os, sys, codecs
import random 


class Dataset():
    def __init__(self, id_data, batch_size, num_steps, target_num_step=None, pad_id=1, target_null_id=0, deterministic=False):
        self.data            = id_data       # it should be id-based data
        
        self.token_pad_id    = pad_id
        self.target_null_id  = target_null_id

        self.batch_size      = batch_size
        self.num_steps       = num_steps

        self.src_num_steps   = num_steps
        self.tar_num_steps   = target_num_step  # for sequence to sequence dataset

        self.deterministic   = deterministic  # if deterministic is True, data is shuffled and retrieved
        self.iterator           = self.iterate_forever()
        self.predict_iterator   = self.iterate_once()

        self.epoch = 0 

    def get_num_examples(self): return len( self.data ) 
    def get_epoch_num(self): return self.epoch 

    def _iterate(self, index_gen, batch_size, max_len):
        """ Abstraction method for _iterate function"""
        raise NotImplementedError("Abstract method.".format( self.run.__name__))
    
    # for training
    def iterate_forever(self):
        def index_stream():
            # yield data index 
            self.indexs = list( range( self.get_num_examples() ) )
            while True:
                self.epoch += 1 
                if not self.deterministic:
                    random.shuffle( self.indexs ) 
                for index in self.indexs:
                    yield index 

        for a_data in self._iterate(index_stream()):
            yield a_data

    # for testing
    def iterate_once(self):
        def index_stream():
            # yield data index 
            self.indexs = list( range( self.get_num_examples() ) )
            for index in self.indexs:
                yield index 

        for a_data in self._iterate(index_stream()):
            yield a_data
