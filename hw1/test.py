#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy
from model_Daikon import RNN_model
import config

def load_data(mode):
  path = ''
  if mode == 'train':
    path = './data/trainframes.npy'
    file = open('./data/labels.npy', 'rb')
    labels = np.load(file)
    file.close()
  elif mode == 'test':
    path = './data/testframes.npy'
    labels = None

  file = open(path, 'rb')
  frames = np.load(file)
  file.close()

  print('Data loaded.')
  return frames, labels

args = config.parse_arguments()
args.fac = int(args.use_bidirection) + 1


with tf.Graph().as_default():
  with tf.name_scope('test'):
    test_args = copy.deepcopy(args)
    test_args.mode = 'test'
    test_args.batch_size = 1
  with tf.variable_scope('model', reuse=None):
    test_model = RNN_model(args=test_args)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.graph_options.optimizer_options.global_jit_level =\
    tf.OptimizerOptions.ON_1
  sv = tf.train.Supervisor(logdir=args.log_dir)

  test_frames, _ = load_data('test')
  with sv.managed_session(config=config) as sess:
    global_step = sess.run(test_model.step)
    print('global step = {}'.format(global_step))

    batch_frames = np.expand_dims(test_frames[:args.window_size, :], 0)
    feed_dict = {test_model.frames_holder: batch_frames}
    prediction = sess.run(test_model.pred, feed_dict=feed_dict)
    print(prediction.shape)
    print(prediction[0])