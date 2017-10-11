#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy
from model import RNN_model
import config

def run_epoch(sess, model):
  '''Runs the model for one epoch'''
  fetches = {}
  fetches['loss'] = model.loss
  if model.is_train():
    fetches['eval'] = model.eval
  vals = sess.run(fetches)
  return vals['loss']

if __name__ == '__main__':
  args = config.parse_arguments()
  args.fac = int(args.use_bidirection) + 1

  with tf.Graph().as_default() as graph:
    initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
    with tf.name_scope('train'):
      train_args = copy.deepcopy(args)
      train_args.mode = 'train'

      with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
        train_model = RNN_model(args=train_args)
        scope.reuse_variables()


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level =\
      tf.OptimizerOptions.ON_1
    sv = tf.train.Supervisor(logdir=args.log_dir,
                             save_model_secs=args.save_model_secs)

    with sv.managed_session(config=config) as sess:
      global_step = sess.run(train_model.step)
      for i in range(global_step+1, args.max_epoch+1):
        train_loss = run_epoch(sess, train_model)
        if i % args.info_epoch == 0:
          print('Epoch: %d Training Loss: %.5f'%(i, train_loss))
