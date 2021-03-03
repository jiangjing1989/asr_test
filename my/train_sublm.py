#!/home/liaochunping/anaconda3/envs/asr/bin/python
import os
import sys
import numpy as np
import argparse
import logging

import tensorflow as tf
from utils.user_config import UserConfig
from LM_model.TransducerPrediction import TransducerPrediction as LM
from LM_classifier.classifier import LM_Classifier
from dataloaders.text_dataloader import Text_Dataloader

os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

class Sub_LM_Trainer():
    def __init__(self, config):
        self.config = config
        self.dg = Text_Dataloader(config['dataloader'])
        self.model = LM(**config['model'])
        self.classifier = LM_Classifier(**config['classifier'])
        self.strategy = tf.distribute.MirroredStrategy()
        self.optimizer = tf.keras.optimizers.Adamax(**config['optimizer'])
        self.loss_history = []
        self._set_train_metrics()

    def _set_train_metrics(self):
        self.train_metrics = {"loss": tf.keras.metrics.Mean("loss", dtype=tf.float32)}

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        x, y = batch
        print(x)
        print(y)
        with tf.GradientTape() as tape:
            print('**********************************************5**********************************************************')       
            hidden_feature = self.model(x, training=True)
            print('**********************************************6**********************************************************')       
            output = self.classifier(hidden_feature)
            per_train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y) 
            train_loss = tf.nn.compute_average_loss(per_train_loss,global_batch_size=self.config['batch'])
        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_metrics['loss'].update_state(per_train_loss)

    def _save_model(self, epoch, loss):
        max_num = self.config['model_save_num']
        model_dir = self.config['model_dir']
        tag = False
        if len(self.loss_history) < max_num:
            self.loss_history.append([epoch, loss])
            tag = True
        else:
            self.loss_history.sort(key=lambda x:x[1])
            if self.loss_history[-1][1] > loss:
                os.remove(os.path.join(model_dir, 'model_{}.h5'.format(self.loss_history[-1][0])))
                del self.loss_history[-1]
                self.loss_history.append([epoch, loss])
                tag = True
        if tag:
            self.model.save_weights(os.path.join(model_dir, 'model_{}.h5'.format(epoch)))

    def train(self):
        train_datasets = tf.data.Dataset.from_generator(self.dg.generator, self.dg.return_data_types(), self.dg.return_data_shape(), args=(True,))
        eval_datasets = tf.data.Dataset.from_generator(self.dg.generator, self.dg.return_data_types(), self.dg.return_data_shape(), args=(False,))
        print('**********************************************1***********************************************************')       
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_datasets = train_datasets.with_options(options)
        eval_datasets = eval_datasets.with_options(options)
        train_datasets=self.strategy.experimental_distribute_dataset(train_datasets)
        eval_datasets=self.strategy.experimental_distribute_dataset(eval_datasets)
        print('**********************************************2***********************************************************')       

        for epoch in range(self.config['epoch']):
            old_epoch = self.dg.epochs
            for batch in train_datasets:
                print('**********************************************3**********************************************************')       
                try:
                    self.strategy.run(self._train_step,args=(batch,))
                except tf.errors.OutOfRangeError:
                    continue
                new_epoch = self.dg.epochs
                print('**********************************************4**********************************************************')       
                if new_epoch - old_epoch >= 1:
                    break
            loss = self.train_metrics['loss'].result().numpy()
            self._save_model(epoch, loss)
            print('epoch:{} loss:{}'.format(epoch, loss))
            self.train_metrics['loss'].reset_states()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--common_config', type=str, default='/home/jiangjing/project/my/config/common.yml')
    args=parse.parse_args()
    config = UserConfig(args.common_config)
    train = Sub_LM_Trainer(config)
    train.train()
