import tensorflow as tf
from tensorflow.contrib import keras
import random
import numpy as np

class RNN_model():
  def __init__(self, args):
    self.__dict__ = args.__dict__.copy()
    rnn_cells = self.initialize()
    self._step = tf.contrib.framework.get_or_create_global_step()
    if self.is_train():
      with tf.variable_scope('build'):
        batch_frames, batch_labels = self.get_batch()
        self._loss = self.build_model(rnn_cells, batch_frames, batch_labels)
        self._loss /= self.batch_size
      self._eval = self.optimize(self._loss)
      _vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      for _var in _vars:
        name = '  '.join('{}'.format(v) for v in _var.name.split('/'))
        print('{:85} {}'.format(name, _var.get_shape()))
    elif self.is_test():
      with tf.variable_scope('build'):
        pass
    else: # validation
      with tf.variable_scope('build'):
        pass

  def get_pred(self, rnn_cells, feed_frames):
    f1_cells = rnn_cells(self.hidden_size)
    if self.use_bidirection:
      b1_cells = rnn_cells(self.hidden_size)
    else:
      b1_cells = None

    outputs, _ = self.rnn_output(feed_frames, f1_cells, b1_cells, 'rnn_model')

    flatten_outputs = tf.reshape(outputs, [-1, self.hidden_size])
    dense1 = tf.layers.dense(flatten_outputs, 512, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu)
    dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu)
    pred = tf.layers.dense(dense3, self.n_class)
    return pred

  def build_model(self, rnn_cells, frames, labels):
    pred = self.get_pred(rnn_cells, frames)
    if  self.is_train():
      return self.calc_loss(pred, labels)
    elif self.is_test():
      return pred
    else:
      return None

  def initialize(self):
    if self.rnn_type == 0:#LSTM
      def unit_cell(fac):
        return tf.contrib.rnn.LSTMCell(fac, use_peepholes=True)
    elif self.rnn_type == 1:#GRU
      def unit_cell(fac):
        return tf.contrib.rnn.GRUCell(fac)
    rnn_cell = unit_cell
    #dropout layer
    if not self.is_test() and self.keep_prob < 1:
      def rnn_cell(fac):
        return tf.contrib.rnn.DropoutWrapper(unit_cell(fac),
                                             output_keep_prob=self.keep_prob)
    def rnn_cells(fac):
      return tf.contrib.rnn.MultiRNNCell([rnn_cell(fac)
                                          for _ in range(self.rnn_layer_num)])
    return rnn_cells

  def get_batch(self):
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
    frames, labels = load_data(self.mode)

    def get_single_ex(frames, labels):
      start_id = random.randint(0, len(frames) - self.window_size - 1)
      ex_frames = frames[start_id:start_id + self.window_size]
      ex_labels = labels[start_id:start_id + self.window_size]
      return ex_frames, ex_labels

    batch_frames = np.zeros([self.batch_size, self.window_size, self.dim], dtype=float)
    batch_labels = np.zeros([self.batch_size, self.window_size, self.n_class], dtype=float)
    for idx in range(self.batch_size):
      ex_frames, ex_labels = get_single_ex(frames, labels)
      batch_frames[idx] = ex_frames.copy()
      batch_labels[idx] = keras.utils.to_categorical(ex_labels, self.n_class).copy()
    return batch_frames, batch_labels

  def is_train(self): return self.mode == 'train'
  def is_valid(self): return self.mode == 'valid'
  def is_test(self): return self.mode == 'test'

  def rnn_output(self, inp, f_cells, b_cells, scope):
    if self.use_bidirection:
      oup, st = tf.nn.bidirectional_dynamic_rnn(f_cells, b_cells, inp,
                                               dtype=tf.float32, scope=scope)
      oup = tf.concat(oup, -1)
      st = tf.concat(st, -1)
    else:
      oup, st =\
        tf.nn.dynamic_rnn(f_cells, inp, dtype=tf.float32, scope=scope)
    oup =\
      tf.reshape(oup, [self.batch_size, -1, self.fac * f_cells.state_size[0]])
    return oup, st[-1]

  def calc_loss(self, pred, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred)
    return tf.reduce_sum(loss)

  def optimize(self, loss):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
               self.max_grad_norm)
    self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                    self.step,
                                                    self.decay_steps,
                                                    self.decay_rate)
    opt = tf.train.AdamOptimizer(self.learning_rate)
    return opt.apply_gradients(zip(grads, tvars), global_step=self._step)

  @property
  def pred(self): return self._pred
  @property
  def loss(self): return self._loss
  @property
  def eval(self): return self._eval
  @property
  def step(self): return self._step
