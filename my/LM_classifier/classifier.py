import os
import tensorflow as tf


class LM_Classifier(tf.keras.Model):
    def __init__(self,
                 dense_dim: int,
                 vocabulary_size: int):
        super(LM_Classifier, self).__init__()
        self.dense = tf.keras.layers.Dense(dense_dim)
        self.out = tf.keras.layers.Dense(vocabulary_size)

    def call(self, inputs, training=False):
        y = self.dense(inputs, training=training)
        y = self.out(y)
        return y

    def get_config(self):
        conf = super(LM_Classifier, self).get_config()
        conf.update(self.dense.get_config())
        conf.update(self.out.get_config())
        return conf
