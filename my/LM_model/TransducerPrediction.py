import os
import tensorflow as tf


class TransducerPrediction(tf.keras.Model):
    def __init__(self, 
                 vocabulary_size: int,
                 embed_dim: int = 128,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 512,
                 name = "transducer_prediction",
                 **kwargs):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embed_dim, mask_zero=False)
        self.do = tf.keras.layers.Dropout(embed_dropout)
        self.lstm_cells = []
        for i in range(num_lstms):
            lstm = tf.keras.layers.LSTMCell(units=lstm_units,)
            self.lstm_cells.append(lstm)
        self.decoder_lstms = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(self.lstm_cells, name="decoder_lstms"),
                                                 return_sequences=False, return_state=False)
    def get_initial_state(self, input_sample):
        return self.decoder_lstms.get_initial_state(input_sample)

    def call(self, inputs, training=False, p_memory_states=None, **kwargs):
        print('---------------------------call1------------------------------------')
        print(inputs)
        outputs = self.embed(inputs, training=training)
        print('---------------------------call2------------------------------------')
        outputs = self.do(outputs, training=training)
        if p_memory_states is None:
            p_memory_states = self.get_initial_state(outputs)
        outputs = self.decoder_lstms(outputs, training=training, initial_state=p_memory_states)
        #outputs = outputs[0]
        return outputs

    def get_config(self):
        conf = super(TransducerPrediction, self).get_config()
        conf.update(self.embed.get_config())
        conf.update(self.do.get_config())
        for lstm in self.lstms:
            conf.update(lstm.get_config())
        return conf
        
