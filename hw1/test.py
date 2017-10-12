import tensorflow as tf
import config
import numpy as np

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
def model(feed_frames):
  cell = tf.contrib.rnn.GRUCell(num_units=args.hidden_size, scope='rnn')
  init_state = cell.zero_state(args.batch_size, dtype=tf.float32, scope='rnn')
  outputs, _ = tf.nn.dynamic_rnn(cell, feed_frames, initial_state=init_state, time_major=False, scope='rnn')
  flatten_outputs = tf.reshape(outputs, [-1, args.hidden_size], name='flatten_outputs')
  dense1 = tf.layers.dense(flatten_outputs, 512, activation=tf.nn.relu, name='dense1')
  dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu, name='dense2')
  dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu, name='dense3')
  pred = tf.layers.dense(dense3, args.n_class, name='pred')
  return pred

args = config.parse_arguments()
tf.reset_default_graph()


feed_frames = tf.placeholder(tf.float32, [None, args.window_size, args.dim])
feed_labels = tf.placeholder(tf.float32, [None, args.window_size])

pred = model(feed_frames)

test_frames, _ = load_data('test')
print(test_frames.shape)

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, './logs/model.ckpt')
  batch_frames = np.expand_dims(test_frames[:args.window_size], 0)
  print(batch_frames.shape)
  prediction = sess.run([pred], {feed_frames: batch_frames})
  print("prediction.shape = ", prediction.shape)
  print(prediction[:3])