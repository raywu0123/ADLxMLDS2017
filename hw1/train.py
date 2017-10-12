import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib import keras
import config


args = config.parse_arguments()

def load_data(mode='train'):
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
    batch_labels[idx] = ex_labels.copy()
  return batch_frames, batch_labels

def model(feed_frames):
  cell = tf.contrib.rnn.GRUCell(num_units=args.hidden_size)
  init_state = cell.zero_state(args.batch_size, dtype=tf.float32)
  outputs, _ = tf.nn.dynamic_rnn(cell, feed_frames, initial_state=init_state, time_major=False)
  flatten_outputs = tf.reshape(outputs, [-1, args.hidden_size])
  dense1 = tf.layers.dense(flatten_outputs, 512, activation=tf.nn.relu)
  dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu)
  dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu)
  pred = tf.layers.dense(dense3, args.n_class)
  return pred

frames, labels = load_data('train')

feed_frames = tf.placeholder(tf.float32, [None, args.window_size, args.dim])
feed_labels = tf.placeholder(tf.float32, [None, args.window_size, 1])
pred = model(feed_frames)

flatten_labels = tf.reshape(feed_labels, [-1])
one_hot_labels = tf.one_hot(tf.cast(flatten_labels, dtype=tf.int64), args.n_class)
loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=pred)
optimizer = tf.train.AdamOptimizer(args.learning_rate)
train_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(pred, 1)), tf.float32))

sv = tf.train.Supervisor(logdir=args.log_dir, save_model_secs=args.save_model_secs)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = \
  tf.OptimizerOptions.ON_1
with sv.managed_session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(args.max_epoch):
    batch_frames, batch_labels = get_batch(frames, labels)
    sess.run(train_op, feed_dict={feed_frames: batch_frames,
                                  feed_labels: batch_labels})

    if (i + 1) % args.info_epoch == 0:
      print(i, sess.run([loss, accuracy], feed_dict={feed_frames: batch_frames,
                                                     feed_labels: batch_labels}))


