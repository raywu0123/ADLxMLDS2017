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
        self._frames_holder = tf.placeholder(tf.float32, [None, self.window_size, self.dim])
        self._labels_holder = tf.placeholder(tf.float32, [None, self.n_class])

        self._pred = self.get_pred(rnn_cells, self._frames_holder)
        self._loss = self.calc_loss(self._pred, self._labels_holder)
        self._acc = self.calc_acc(self._pred, self._labels_holder)

      self._eval = self.optimize(self._loss)
      _vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      for _var in _vars:
        name = '  '.join('{}'.format(v) for v in _var.name.split('/'))
        print('{:85} {}'.format(name, _var.get_shape()))
    elif self.is_test():
      with tf.variable_scope('build'):
        self._frames_holder = tf.placeholder(tf.float32, [None, self.window_size, self.dim])
        self._labels_holder = tf.placeholder(tf.float32, [None, self.n_class])
        self._pred = self.get_pred(rnn_cells, self._frames_holder)
    else:  # validation
      with tf.variable_scope('build'):
        pass

  def get_pred(self, rnn_cells, feed_frames):

    with tf.variable_scope('RNN_1'):
      f1_rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
      if self.use_bidirection:
        b1_rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
      else:
        b1_rnn_cell = None
      outputs1, _ = self.rnn_output(feed_frames, f1_rnn_cell, b1_rnn_cell, 'rnn_model')

    with tf.variable_scope('RNN_2'):
      f2_rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size//2)
      if self.use_bidirection:
        b2_rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size//2)
      else:
        b2_rnn_cell = None
      outputs2, _ = self.rnn_output(outputs1, f2_rnn_cell, b2_rnn_cell, 'rnn_model')

    flatten_outputs = tf.reshape(outputs2, [-1, self.hidden_size//2*(1+int(self.use_bidirection))])
    pred = tf.layers.dense(flatten_outputs, self.n_class)
    return pred

  def initialize(self):
    if self.rnn_type == 0:  # LSTM
      def unit_cell(fac):
        return tf.contrib.rnn.LSTMCell(fac, use_peepholes=True)
    elif self.rnn_type == 1:  # GRU
      def unit_cell(fac):
        return tf.contrib.rnn.GRUCell(fac)
    rnn_cell = unit_cell
    # dropout layer
    if not self.is_test() and self.keep_prob < 1:
      def rnn_cell(fac):
        return tf.contrib.rnn.DropoutWrapper(unit_cell(fac),
                                             output_keep_prob=self.keep_prob)

    def rnn_cells(fac):
      return tf.contrib.rnn.MultiRNNCell([rnn_cell(fac)
                                          for _ in range(self.rnn_layer_num)])

    return rnn_cells

  def is_train(self):
    return self.mode == 'train'

  def is_valid(self):
    return self.mode == 'valid'

  def is_test(self):
    return self.mode == 'test'

  def rnn_output(self, inp, f_cells, b_cells, scope):
    seq_len = tf.constant(self.batch_size, dtype=tf.int64, shape=[self.batch_size])
    if self.use_bidirection:
      oup, st = tf.nn.bidirectional_dynamic_rnn(f_cells, b_cells, inp, seq_len,
                                                dtype=tf.float32, scope=scope)
      oup = tf.concat(oup, -1)
      st = tf.concat(st, -1)
    else:
      oup, st = \
        tf.nn.dynamic_rnn(f_cells, inp, seq_len, dtype=tf.float32, scope=scope)
    oup = \
      tf.reshape(oup, [self.batch_size, -1, self.fac * f_cells.state_size[0]])
    return oup, st[-1]

  def calc_loss(self, pred, labels):
    return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred)

  def calc_acc(self, pred, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(pred, 1)), tf.float32))

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
  def loss(self): return self._loss

  @property
  def acc(self):  return self._acc

  @property
  def eval(self): return self._eval

  @property
  def step(self):
    return self._step

  @property
  def frames_holder(self):  return self._frames_holder

  @property
  def labels_holder(self):  return self._labels_holder

  @property
  def pred(self): return self._pred

class CNN_model(RNN_model):
  def get_pred(self, rnn_cells, feed_frames):

    reshape1 = tf.expand_dims(feed_frames, -1)
    conv1 = tf.layers.conv2d(reshape1, self.filter_num,
                               [5, 3],
                               activation=tf.nn.relu, padding='SAME')
    pool1 = tf.layers.max_pooling2d(conv1, 2, [2, 2], padding='valid')

    reshape2 = tf.reshape(pool1, [self.batch_size, self.window_size, (self.dim//2)*self.filter_num//2])


    with tf.variable_scope('RNN_1'):
      f1_cells = rnn_cells(self.hidden_size)
      if self.use_bidirection:
        b1_cells = rnn_cells(self.hidden_size)
      else:
        b1_cells = None
      outputs1, _ = self.rnn_output(reshape2, f1_cells, b1_cells, 'cnn_model')

    with tf.variable_scope('RNN_2'):
      f1_cells = rnn_cells(self.hidden_size//2)
      if self.use_bidirection:
        b1_cells = rnn_cells(self.hidden_size//2)
      else:
        b1_cells = None
      outputs2, _ = self.rnn_output(outputs1, f1_cells, b1_cells, 'cnn_model')

    flatten_outputs = tf.reshape(outputs2, [-1, self.hidden_size//2*(1+int(self.use_bidirection))])
    pred = tf.layers.dense(flatten_outputs, self.n_class)
    return pred