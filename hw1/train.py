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
    batch_labels[idx] = keras.utils.to_categorical(ex_labels, args.n_class).copy()
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
feed_labels = tf.placeholder(tf.float32, [None, args.window_size, args.n_class])
flatten_labels = tf.reshape(feed_labels, [-1, args.n_class])
pred = model(feed_frames)

loss = tf.losses.softmax_cross_entropy(onehot_labels=flatten_labels, logits=pred)
optimizer = tf.train.AdamOptimizer(args.learning_rate)
train_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(flatten_labels, 1), tf.argmax(pred, 1)), tf.float32))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in range(args.max_epoch):
    batch_frames, batch_labels = get_batch(frames, labels)
    sess.run(train_op, feed_dict={feed_frames: batch_frames,
                                  feed_labels: batch_labels})

    if (i + 1) % args.info_epoch == 0:
      print(i, sess.run([loss, accuracy], feed_dict={feed_frames: batch_frames,
                                                     feed_labels: batch_labels}))

test_frames, _ = load_data('test')
print(test_frames.shape)
