#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy
from model_Daikon import RNN_model
import config
import math
from tqdm import tqdm

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
  with tf.variable_scope('model', reuse=None):
    test_model = RNN_model(args=test_args)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.graph_options.optimizer_options.global_jit_level =\
    tf.OptimizerOptions.ON_1
  sv = tf.train.Supervisor(logdir=args.log_dir)

  test_frames, _ = load_data('test')

  with sv.managed_session() as sess:
    # global_step = sess.run(test_model.step)
    # print('global step = {}'.format(global_step))
    frame_scores = np.zeros([test_frames.shape[0], args.n_class], dtype=float)
    n_frames = test_frames.shape[0]
    for frame_id in tqdm(range(n_frames)):
      batch_frames = np.zeros([args.batch_size, args.window_size, args.dim])
      batch_frames[frame_id % args.window_size, :, :] = test_frames.take(
        range(frame_id - args.window_size//2, frame_id + args.window_size//2),
        mode='wrap', axis=0)

      if (frame_id + 1) % args.batch_size or frame_id == n_frames-1:
        feed_dict = {test_model.frames_holder: batch_frames}
        prediction = sess.run(test_model.pred, feed_dict=feed_dict)
        frame_scores[frame_id-args.batch_size+1:frame_id+1] = prediction

    with open('./prediction.npy','wb+') as file:
      np.save(file, frame_scores)
      print('Prediction saved to {}') % format(file.name)