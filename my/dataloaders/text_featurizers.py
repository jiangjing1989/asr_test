import os
import sys
import codecs
import tensorflow as tf

class TextFeaturizer:
    def __init__(self, decoder_config:dict):
        self.decoder_config = decoder_config
        self.scores = None
        self.num_classes = 0
        lines = []
        with codecs.open(self.decoder_config['vocabulary'], 'r', 'utf-8') as f:
            lines.extend(f.readlines())
        
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_array = []
        self.tf_vocab_array = tf.constant([], dtype = tf.string)
        self.index_to_unicode_points = tf.constant([], dtype = tf.int32)

        if self.decoder_config['blank_at_zero']:
            self.blank = 0
            index = 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, ['']], axis = 0)
            self.index_to_unicode_points = tf.concat([self.index_to_unicode_points, [0]], axis = 0)
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line or line == '\n':
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.vocab_array.append(line)
            self.tf_vocab_array = tf.concat([sekf.tf_vocab_array, [line]], axis=0)
            upoint = tf.strings.unicode_decode(line, 'utf-8')
            self.index_to_unicode_points = tf.concat([self.index_to_unicode_points, upoint], axis=0)
            index += 1
        self.num_classes = index
        if not self.decoder_config['blank_at_zero']:
            self.blank = index
            self.num_classes += 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, ['']], axis = 0)
            self.index_to_unicode_points = tf.concat([self.index_to_unicode_points, [0]], axis = 0)
        self.stop = self.endid()
        self.pad = self.blank
        self.start = self.startid()

    def add_scorer(self, scorer: any = None):
        self.scorer = scorer

    def startid(self):
        return self.token_to_index['S']

    def endid(self):
        return self.token_to_index['/S']

    def prepand_blank(self, text:tf.Tensor) -> tf.Tensor:
        return tf.concat([[self.blank], text], axis=0)

    def extract(self, tokens):
        new_tokens = []
        for tok in tokens:
            if tok in self.vocab_array:
                new_tokens.append(tok)
            else:
                print(tok, ' not in vocab_array')
                continue
        tokens = new_tokens
        feats = [self.start] + [self.token_to_index[token] for token in tokens]
        return feats

    @tf.function
    def iextract(self, feat):
        with tf.name_scope('invert_text_extraction'):
            minus_one = -1 * tf.ones_like(feat, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(feat, dtype=tf.int32)
            feat = tf.where(feat == minus_one, blank_like, feat)
            return tf.map_fn(self._idx_to_char, feat, dtype=tf.string)

    def index2upoints(self, feat:tf.Tensor) -> tf.Tensor:
        with tf.name_scope('index_to_unicode_points'):
            def map_fn(arr):
                def sub_map_fn(index):
                    return self.index_to_unicode_points[index]
                return tf.map_fn(sub_map_fn, arr, dtype=tf.int32)
            minus_one = -1 * tf.ones_like(feat, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(feat, dtype=tf.int32)
            feat = tf.where(feat == minus_one, blank_like, feat)
            return tf.map_fn(map_fn, feat, dtype=tf.int32)
