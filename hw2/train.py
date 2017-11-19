#!/usr/bin/python3

import config
import random
import tensorflow as tf
import pickle
import copy
import numpy as np
import os
from model import S2VT_model, seq2seq_model
from utils import calc_bleu, int2string, get_inference_batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
args = config.parse_arguments()
args.fac = int(args.use_bidirection) + 1

dct = open(os.path.join(args.preprocess_dir, 'vocab.txt'), 'r', encoding='utf8').read().splitlines()
args.vocab_size = len(dct)

video_ids = []
with open(os.path.join(args.data_dir, 'testing_id.txt')) as file:
  for line in file:
    video_ids.append(line.strip('\n'))


def load_data(mode='training'):
  pkl_file = open(os.path.join(args.preprocess_dir, mode+'_int_captions.pkl'), 'rb')
  captions = pickle.load(pkl_file)
  pkl_file.close()
  print('Captions Loaded.')
  return captions

def run_epoch(sess, model, args, captions, opt):
  # batch_vgg.shape = [batch_size, frame_num, feat_num]
  # batch_captions.shape = [batch_size, max_sent_len]

  def get_batch(captions):
    def aug_vgg(vgg, ratio):
      Rshift_vgg = vgg.take(range(1, args.frame_num+1), mode='wrap', axis=0)
      Lshift_vgg = vgg.take(range(-1, args.frame_num-1), mode='wrap', axis=0)
      probs = np.random.choice([1.0, 1/3], size=args.frame_num, p=[1-ratio, ratio])

      aug = (vgg.T*probs).T + (Rshift_vgg.T*((1-probs)/2)).T + (Lshift_vgg.T*((1-probs)/2)).T
      return aug


    dir_list = list(captions.keys())
    batch_vggs = np.zeros([args.batch_size, args.frame_num, args.feat_num])
    batch_captions = np.zeros([args.batch_size, args.max_sent_len], dtype=int)
    batch_lens = np.zeros([args.batch_size], dtype=int)

    for idx in range(args.batch_size):
      video_name = random.choice(dir_list)
      (caption, length) = random.choice(captions[video_name])
      vgg = np.load(os.path.join(args.data_dir, 'training_data/feat/', video_name + '.npy'))
      batch_captions[idx] = caption
      if args.aug:
        batch_vggs[idx] = aug_vgg(vgg, 0.25)
      else:
        batch_vggs[idx] = vgg
      batch_lens[idx] = length

    return batch_vggs, batch_captions, batch_lens

  batch_vggs, batch_captions, batch_lens = get_batch(captions)
  fetches = {}
  fetches['loss'] = model.loss
  fetches['pred'] = model.pred
  feed_dict = {model.video_holder: batch_vggs,
               model.caption_holder: batch_captions,
               model.len_holder: batch_lens}
  if opt:
    fetches['eval'] = model.eval
  else:
    model.sche_prob = 0.0

  vals = sess.run(fetches, feed_dict=feed_dict)
  pred = vals['pred']
  example = (pred[50], batch_captions[50][1:])

  return vals['loss'], example

def run_inference(sess, test_model, batch, special_id=()):
  batch_vggs, batch_captions, batch_lens = batch

  feed_dict = {test_model.video_holder: batch_vggs,
               test_model.caption_holder: batch_captions,
               test_model.len_holder: batch_lens}
  pred = sess.run(test_model.pred, feed_dict=feed_dict)

  with open(args.output_file, 'w+') as file:
    for idx, video_name in enumerate(video_ids):
      if len(special_id) == 0 or (len(special_id) != 0 and video_name in special_id):
        line = video_name + ',' + int2string(pred[idx]) + '\n'
        file.write(line)

  return pred

if __name__ == '__main__':
  with tf.Graph().as_default() as graph:
    initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
    with tf.name_scope('train'):
      train_args = copy.deepcopy(args)
      train_args.mode = 'train'

    with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
      if train_args.model_type == 'S2VT':
        train_model = S2VT_model(args=train_args)
      elif train_args.model_type == 'seq2seq':
        train_model = seq2seq_model(train_args)
      scope.reuse_variables()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level =\
      tf.OptimizerOptions.ON_1
    sv = tf.train.Supervisor(logdir=args.log_dir,
                             save_model_secs=args.save_model_secs)

    with sv.managed_session(config=config) as sess:
      wordvec = np.load(os.path.join(args.preprocess_dir, 'my_wv.npy'))
      sess.run(train_model.embed_init, feed_dict={train_model.embed: wordvec})
      train_captions = load_data('training')
      global_step = sess.run(train_model.step)
      print('global step = {}'.format(global_step))
      inference_batch = get_inference_batch(video_ids)

      train_loss_rec = []
      test_bleu_rec = []
      for i in range(global_step+1, args.max_epoch+1):
        train_loss, example = run_epoch(sess, train_model, train_args, train_captions, opt=True)
        print(int2string(example[0]))
        print(int2string(example[1]))
        train_loss_rec.append(train_loss)
        if i % args.info_epoch == 0:
          print('Epoch: %d, TrainLoss: %.5f' % (i, train_loss))
          train_model.set_mode('test')
          pred = run_inference(sess, train_model, inference_batch)
          print(int2string(pred[0]))
          scores = calc_bleu(args.output_file)
          test_bleu_rec.append(scores)
          print('bleu: ', scores)
          train_model.set_mode('train')

      np.save('train_loss.npy', np.array(train_loss_rec))
      np.save('test_bleu.npy', np.array(test_bleu_rec))
      print('Record npy files saved.')