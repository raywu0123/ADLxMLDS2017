#!/usr/bin/python3
import config
import random
import tensorflow as tf
import pickle
import copy
import numpy as np
import os
from model import S2VT_model, seq2seq_model
from utils import calc_bleu, get_inference_batch
from train import run_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
args = config.parse_arguments()
args.fac = int(args.use_bidirection) + 1
args.test_num = 100

dct = open(os.path.join(args.preprocess_dir, 'vocab.txt'), 'r', encoding='utf8').read().splitlines()
args.vocab_size = len(dct)
video_ids = []

special_ids = ['klteYv1Uv9A_27_33.avi',
               '5YJaS2Eswg0_22_26.avi',
               'UbmZAe5u5FI_132_141.avi',
               'JntMAcTlOF0_50_70.avi',
               'tJHUH9tpqPg_113_118.avi']

video_id_file = args.test_mode + '_id.txt'
with open(os.path.join(args.data_dir, video_id_file)) as file:
  for line in file:
    video_ids.append(line.strip('\n'))
args.batch_size = len(video_ids)

if __name__ == '__main__':
  with tf.Graph().as_default() as graph:
    with tf.name_scope('test'):
      test_args = copy.deepcopy(args)
      test_args.mode = 'test'

    with tf.variable_scope('model', reuse=None) as scope:
      if test_args.model_type == 'S2VT':
        test_model = S2VT_model(args=test_args)
      elif test_args.model_type == 'seq2seq':
        test_model = seq2seq_model(test_args)

    inference_batch = get_inference_batch(video_ids)
    sv = tf.train.Supervisor(logdir=args.log_dir)
    with sv.managed_session() as sess:
      global_step = sess.run(test_model.step)
      print('global step = {}'.format(global_step))
      pred = run_inference(sess, test_model, inference_batch, special_ids)
