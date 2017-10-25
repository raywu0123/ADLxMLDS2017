#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy
from model_Daikon import RNN_model, CNN_model
import config
import random
from os.path import join
from tensorflow.contrib import keras

def load_data(mode):
  path = ''
  if mode == 'train':
    path = join(args.data_dir, 'trainframes.npy')
    file = open(join(args.data_dir, 'labels.npy'), 'rb')
    labels = np.load(file)
    file.close()
  elif mode == 'test':
    path = join(args.data_dir, 'testframes.npy')
    labels = None

  file = open(path, 'rb')
  frames = np.load(file)
  file.close()

  print('Data loaded.')
  return frames, labels

def run_epoch(sess, model, args, frames, labels, opt):
  def get_batch(frames, labels):
    def get_single_ex(frames, labels):
      start_id = random.randint(0, len(frames) - args.window_size - 1)
      ex_frames = frames[start_id:start_id + args.window_size]
      ex_labels = labels[start_id:start_id + args.window_size]
      return ex_frames, ex_labels

    batch_frames = np.zeros([args.batch_size, args.window_size, args.dim], dtype=float)
    batch_labels = np.zeros([args.batch_size, args.window_size, args.n_class], dtype=float)
    for idx in range(args.batch_size):
      ex_frames, ex_labels = get_single_ex(frames, labels)
      batch_frames[idx] = ex_frames.copy()
      batch_labels[idx] = keras.utils.to_categorical(ex_labels, args.n_class).copy()
    return batch_frames, batch_labels
  '''Runs the model for one epoch'''
  batch_frames, batch_labels = get_batch(frames, labels)
  flat_batch_labels = np.reshape(batch_labels, [-1, args.n_class])
  fetches = {}
  fetches['loss'] = model.loss
  fetches['acc'] = model.acc
  feed_dict = {model.frames_holder: batch_frames, model.labels_holder: flat_batch_labels}
  if opt:
    fetches['eval'] = model.eval
  vals = sess.run(fetches, feed_dict=feed_dict)
  return vals['loss'], vals['acc']

if __name__ == '__main__':
  args = config.parse_arguments()
  args.fac = int(args.use_bidirection) + 1

  with tf.Graph().as_default() as graph:
    initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
    with tf.name_scope('train'):
      train_args = copy.deepcopy(args)
      train_args.mode = 'train'

      with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
        if args.model_type == 'RNN':
          train_model = RNN_model(args=train_args)
        elif args.model_type == 'CNN':
          train_model = CNN_model(args=train_args)
        scope.reuse_variables()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level =\
      tf.OptimizerOptions.ON_1
    sv = tf.train.Supervisor(logdir=args.log_dir,
                             save_model_secs=args.save_model_secs)

    frames, labels = load_data(train_args.mode)
    def split(nparray, ratio):
      split_idx = int(nparray.shape[0]*(1-ratio))
      return nparray[:split_idx], nparray[split_idx:]
    train_frames, valid_frames = split(frames, args.valid_ratio)
    train_labels, valid_labels = split(labels, args.valid_ratio)

    with sv.managed_session(config=config) as sess:
      global_step = sess.run(train_model.step)
      for i in range(global_step+1, args.max_epoch+1):
        train_loss, train_acc = run_epoch(sess, train_model, train_args, train_frames, train_labels, opt=True)
        if i % args.info_epoch == 0:
          valid_loss, valid_acc = run_epoch(sess, train_model, train_args, valid_frames, valid_labels, opt=False)
          print('Epoch: %d, TrainLoss: %.5f, TrainAcc= %.5f, ValidLoss: %.5f, ValidAcc= %.5f'\
                  % (i, train_loss, train_acc, valid_loss, valid_acc))
