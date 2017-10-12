import tensorflow as tf
import config

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
  cell = tf.contrib.rnn.GRUCell(num_units=args.hidden_size)
  init_state = cell.zero_state(args.batch_size, dtype=tf.float32)
  outputs, _ = tf.nn.dynamic_rnn(cell, feed_frames, initial_state=init_state, time_major=False)
  flatten_outputs = tf.reshape(outputs, [-1, args.hidden_size])
  dense1 = tf.layers.dense(flatten_outputs, 512, activation=tf.nn.relu)
  dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu)
  dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu)
  pred = tf.layers.dense(dense3, args.n_class)
  return pred

args = config.parse_arguments()

feed_frames = tf.placeholder(tf.float32, [None, args.window_size, args.dim])
feed_labels = tf.placeholder(tf.float32, [None, args.window_size, 1])
flatten_labels = tf.reshape(feed_labels, [-1, 1])
pred = model(feed_frames)

test_frames, _ = load_data('test')
print(test_frames.shape)

sv = tf.train.Supervisor(logdir=args.log_dir)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = \
  tf.OptimizerOptions.ON_1
with sv.managed_session(config=config) as sess:
  batch_frames = test_frames[:args.batch_size]
  prediction = sess.run([pred], {feed_frames: batch_frames})
  print("prediction.shape = ", prediction.shape)
  print(prediction[:3])