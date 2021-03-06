#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy
from model_Daikon import RNN_model, CNN_model
import config
from os.path import join
import math
from tqdm import tqdm

args = config.parse_arguments()
args.fac = int(args.use_bidirection) + 1



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

def write_result(frame_scores, path):
  def get_intchar_map(path):
    phone2int = {}
    phone2char = {}
    with open(path) as file:
      for line in file:
        sep = line[:-1].split('\t')
        phone2int[sep[0]] = int(sep[1])
        phone2char[sep[0]] = str(sep[2])
    return phone2int, phone2char

  def get_phone_map(path):
    phone_map = {}
    with open(path) as file:
      for line in file:
        sep = line[:-1].split('\t')
        phone_map[sep[0]] = str(sep[1])
    return phone_map
  phone2int, phone2char = get_intchar_map(join(args.data_dir, '48phone_char.map'))
  int2phone = {v: k for k, v in phone2int.items()}
  phone_map = get_phone_map(join(args.data_dir, 'phones/48_39.map'))

  split_videos=[]
  ids = []
  with open(join(args.data_dir, 'mfcc/test.ark')) as file:
    pre_id = ''
    video = []
    for line_id, line in enumerate(file):
      line = line.strip('\n').split(' ')
      id = '_'.join(line[0].split('_')[:2])
      if pre_id != id and len(video) != 0:
        split_videos.append(video)
        ids.append(pre_id)
        video = []
      pre_id = id
      video.append(frame_scores[line_id].argmax())
    split_videos.append(video)
    ids.append(pre_id)

  with open(path, 'w+') as file:
    file.write('id,phone_sequence\n')
    for video_id, sequence_48 in enumerate(split_videos):
      sequence_39 = [phone_map[int2phone[integer]] for integer in sequence_48]
      def comb(a, thres=3):
        assert (len(a) > 0)

        pre_char = a[0]
        cnt = 0
        filtered_a = []
        for entry in a:
          cur_char = entry
          if pre_char == cur_char:
            cnt += 1
          else:
            if cnt >= thres:
              filtered_a.append(pre_char)
            pre_char = cur_char
            cnt = 1
        if cnt >= thres:
          filtered_a.append(a[-1])
        return filtered_a
      comb_phone_sequence = comb(comb(sequence_39), 1)
      char_sequence = [phone2char[phone] for phone in comb_phone_sequence]

      string = ''.join(char_sequence).strip('L')
      line = ids[video_id]+","+string
      print(line)
      # input()
      file.write(line+'\n')

with tf.Graph().as_default():
  with tf.name_scope('test'):
    test_args = copy.deepcopy(args)
    test_args.mode = 'test'
  with tf.variable_scope('model', reuse=None):
    if args.model_type == 'RNN':
      test_model = RNN_model(args=test_args)
    elif args.model_type == 'CNN':
      test_model = CNN_model(args=test_args)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.graph_options.optimizer_options.global_jit_level =\
    tf.OptimizerOptions.ON_1
  sv = tf.train.Supervisor(logdir=args.log_dir)

  test_frames, _ = load_data('test')

  with sv.managed_session() as sess:
    global_step = sess.run(test_model.step)
    print('global step = {}'.format(global_step))
    frame_scores = np.zeros([test_frames.shape[0], args.n_class], dtype=float)
    n_frames = test_frames.shape[0]
    n_batch = math.ceil(n_frames/args.batch_size)
    for batch_id in tqdm(range(n_batch)):
      def get_batch(frames, start_id):
        batch_frames = np.zeros([args.batch_size, args.window_size, args.dim])
        for idx in range(args.batch_size):
          batch_frames[idx, :, :] = frames.take(
          range(start_id+idx, start_id+idx + args.window_size),
          mode='wrap', axis=0).copy()
        return batch_frames
      start_id = batch_id*args.batch_size
      batch_frames = get_batch(test_frames, start_id)
      feed_dict = {test_model.frames_holder: batch_frames}
      prediction = sess.run(test_model.pred, feed_dict=feed_dict)
      for pred_batch_idx in range(args.batch_size):
        batch_start_id = pred_batch_idx*args.window_size
        batch_end_id = batch_start_id + args.window_size
        batch_pred = prediction[batch_start_id:batch_end_id]

        def filter_batch_pred(batch_pred, mode):
          filter_pred = np.zeros([args.window_size, args.n_class], dtype=float)
          if mode == 'filter':
            for idx, frame_pred in enumerate(batch_pred):
              def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)
              softmax_frame_pred = softmax(frame_pred)
              n_over_threshold = np.sum(softmax_frame_pred > 0.5)
              # print(n_over_threshold)
              if n_over_threshold > 0 and n_over_threshold < 3:
                filter_pred[idx] = frame_pred.copy()
            return filter_pred
          elif mode == 'strip':
            filter_pred[args.vote_num: -args.vote_num] = batch_pred[args.vote_num: -args.vote_num]
            return filter_pred
        batch_pred = filter_batch_pred(batch_pred, 'strip')

        # input()
        frame_start_id = start_id+pred_batch_idx
        frame_end_id = frame_start_id + args.window_size
        if frame_end_id <= n_frames:
          frame_scores[frame_start_id:frame_end_id]\
            += batch_pred
        elif frame_end_id > n_frames and frame_start_id < n_frames:
          frame_scores[frame_start_id:] += batch_pred[:n_frames-frame_start_id]
          frame_scores[:frame_end_id-n_frames] += batch_pred[n_frames-frame_start_id:]


    write_result(frame_scores, args.pred_file)