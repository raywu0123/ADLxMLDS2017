import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
import math
class S2VT_model():
  def __init__(self, args):
    self.__dict__ = args.__dict__.copy()
    rnn_cells, W_E = self.initialize()
    self.total_len = self.frame_num + self.max_sent_len - 1
    self._step = tf.contrib.framework.get_or_create_global_step()

    with tf.variable_scope('build'):
      self._video_holder = tf.placeholder(tf.float32, [None, self.frame_num, self.feat_num])
      self._caption_holder = tf.placeholder(tf.int32, [None, self.max_sent_len])
      self._len_holder = tf.placeholder(tf.int32, [None])
      self._logits = self.get_logits(rnn_cells, W_E)
      self._pred = tf.argmax(self._logits, 2)
      self._loss = self.calc_loss()
      self._eval = self.optimize()


    if self.is_train():
      _vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      for _var in _vars:
        name = '  '.join('{}'.format(v) for v in _var.name.split('/'))
        print('{:85} {}'.format(name, _var.get_shape()))

  def get_logits(self, rnn_cells, W_E):
    with tf.variable_scope('encoder'):
      enc_outputs = self.encoder(rnn_cells)
    with tf.variable_scope('decoder'):
      logits = self.decoder(rnn_cells, enc_outputs, W_E)

    return logits

  def encoder(self, rnn_cells):
    flat_video = tf.reshape(self._video_holder, [-1, self.feat_num])
    video_emb = tf.layers.dense(flat_video, self.video_emb_dim, tf.nn.relu)
    video_emb = tf.reshape(video_emb, [-1, self.frame_num, self.video_emb_dim])

    enc_padding = tf.zeros([self.batch_size, self.max_sent_len - 1, self.video_emb_dim])
    enc_input = tf.concat([video_emb, enc_padding], 1)

    enc_cells = rnn_cells(self.enc_dim)
    seq_len = tf.constant(self.total_len, dtype=tf.int32, shape=[self.batch_size])
    enc_outputs, enc_final_state = tf.nn.dynamic_rnn(enc_cells,
                                                     enc_input,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)
    return enc_outputs

  def decoder(self, rnn_cells, dec_inputs, W_E):
    target_input = self._caption_holder[:, :-1]  # start from <BOS>

    with tf.variable_scope('word_decode'):
      W_D = tf.get_variable('word_decode', [self.dec_dim, self.vocab_size])

    dec_padding = tf.zeros([self.batch_size, self.frame_num, self.vocab_emb_dim])
    caption_emb = tf.nn.embedding_lookup(W_E, target_input)
    dec_cells = rnn_cells(self.dec_dim)

    seq_len = tf.constant(self.total_len, dtype=tf.int32, shape=[self.batch_size])

    layer_1_outputs = tf.transpose(dec_inputs, perm=[1, 0, 2])
    layer_1_outputs_ta = tf.TensorArray(dtype=tf.float32, size=self.total_len)
    layer_1_outputs_ta = layer_1_outputs_ta.unstack(layer_1_outputs)
    dec_pad_and_embed = tf.concat([dec_padding, caption_emb], 1)
    dec_inputs = tf.concat([dec_pad_and_embed, dec_inputs], 2)

    dec_inputs = tf.transpose(dec_inputs, perm=[1, 0, 2])  # for time major unstack
    dec_inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.total_len)
    dec_inputs_ta = dec_inputs_ta.unstack(dec_inputs)

    rand = tf.random_uniform([self.total_len], dtype=tf.float32)
    rand_ta = tf.TensorArray(dtype=tf.float32, size=self.total_len)
    rand_ta = rand_ta.unstack(rand)

    schedule_sample_prob = self.get_sche_prob()

    def dec_loop_fn(time, cell_output, cell_state, loop_state):
      def input_fn():
        def normal_feed_in():
          return dec_inputs_ta.read(time)

        def schedule_sample():
          if cell_output is None:
            return tf.zeros([self.batch_size, self.vocab_emb_dim + self.enc_dim], dtype=tf.float32)

          def feed_previous():
            output_logit = tf.matmul(cell_output, W_D)
            prediction = tf.argmax(output_logit, 1)
            prediction_embed = tf.nn.embedding_lookup(W_E, prediction)
            next_input = tf.concat([prediction_embed, layer_1_outputs_ta.read(time)], 1)
            return next_input

          sample = schedule_sample_prob[0, 0] < rand_ta.read(time)
          sample = tf.reduce_all(sample)
          return tf.cond(sample, feed_previous, normal_feed_in)

        start_decoding = (time >= self.frame_num + 1)  # first input should keep to be <BOS>
        start_decoding = tf.reduce_all(start_decoding)
        return tf.cond(start_decoding, schedule_sample, normal_feed_in)

      def zeros():
        return tf.zeros([self.batch_size, self.vocab_emb_dim + self.enc_dim], dtype=tf.float32)

      emit_output = cell_output
      if cell_output is None:  # time == 0
        next_cell_state = dec_cells.zero_state(self.batch_size, dtype=tf.float32)
      else:
        next_cell_state = cell_state
      is_finished = (time >= seq_len)
      finished = tf.reduce_all(is_finished)
      next_input = tf.cond(finished, zeros, input_fn)  # dec_inputs_ta.read(time))
      return is_finished, next_input, next_cell_state, emit_output, loop_state

    dec_outputs_ta, dec_final_state, _ = tf.nn.raw_rnn(dec_cells, dec_loop_fn)
    dec_outputs = dec_outputs_ta.stack()
    dec_outputs = dec_outputs[self.frame_num:, :, :]
    dec_outputs = tf.transpose(dec_outputs, perm=[1, 0, 2])  # batch_size x time x embed_dim

    dec_outputs = tf.reshape(dec_outputs, [-1, self.dec_dim])
    dec_output_logit = tf.matmul(dec_outputs, W_D)
    dec_output_logit = tf.reshape(dec_output_logit, [self.batch_size, -1, self.vocab_size])
    return dec_output_logit

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


    W_E = tf.get_variable('W_E', [self.vocab_size, self.vocab_emb_dim],
                          dtype=tf.float32)
    self.embed = tf.placeholder(tf.float32, [self.vocab_size, self.vocab_emb_dim])
    self.embed_init = W_E.assign(self.embed)

    return rnn_cells, W_E

  def is_train(self):
    return self.mode == 'train'

  def is_valid(self):
    return self.mode == 'valid'

  def is_test(self):
    return self.mode == 'test'

  def rnn_output(self, inp, f_cells, b_cells, scope):
    seq_len = tf.constant(self.frame_num, dtype=tf.int64, shape=[self.batch_size])
    # TODO
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

  def calc_loss(self):
    target_output = self._caption_holder[:, 1:]  # end by <EOS>
    caption_mask = tf.sequence_mask(self._len_holder - 1, self.max_sent_len - 1, dtype=tf.float32)
    losses = seq2seq.sequence_loss(self._logits, target_output, caption_mask)
    return tf.reduce_mean(losses)

  def optimize(self):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),
                                      self.max_grad_norm)
    self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                    self.step,
                                                    self.decay_steps,
                                                    self.decay_rate)
    opt = tf.train.AdamOptimizer(self.learning_rate)
    return opt.apply_gradients(zip(grads, tvars), global_step=self._step)

  def set_mode(self, mode):
    if mode == 'train':
      self.mode = 'train'
    elif mode == 'test':
      self.mode = 'test'
    else:
      assert('Undefined mode!')

  def get_sche_prob(self):
    if not self.is_train():
      schedule_sample_prob = tf.constant(0.0, dtype=tf.float32, shape=[1, 1])
    else:
      float_step = tf.cast(self._step, dtype=tf.float32)
      schedule_sample_prob = tf.reshape(tf.exp(-float_step/self.tao), [1, 1])
    return schedule_sample_prob

  @property
  def loss(self): return self._loss

  @property
  def eval(self): return self._eval

  @property
  def step(self): return self._step

  @property
  def video_holder(self):  return self._video_holder

  @property
  def caption_holder(self):  return self._caption_holder

  @property
  def len_holder(self):  return self._len_holder

  @property
  def pred(self): return self._pred


class seq2seq_model(S2VT_model):
  def get_logits(self, rnn_cells, W_E):
    source_seq_embedded = self._video_holder  # shape=(batch_size, 80, 4096)
    embedding_matrix = W_E

    decoder_input_embedded = tf.nn.embedding_lookup(embedding_matrix, self.caption_holder)  # shape=(2, 4, 5)
    enc_seq_len = tf.constant(self.frame_num, dtype=tf.int32, shape=[self.batch_size])
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      tf.contrib.rnn.LSTMCell(self.enc_dim),
      source_seq_embedded,
      sequence_length=enc_seq_len,
      dtype=tf.float32)


    ## inference = 1.0
    sampling_prob = 1.0 - self.get_sche_prob()[0, 0]
    seq_len = tf.constant(self.max_sent_len, dtype=tf.int32, shape=[self.batch_size])
    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
      decoder_input_embedded,
      seq_len,
      W_E,
      sampling_probability=sampling_prob)

    if True:
      beam_width = 1
    else:
      beam_width = 4
    output_layer = Dense(self.vocab_size)
    dec_cell = rnn_cells(self.enc_dim)
    tiled_encoder_outputs = seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
    attention_mechanism = seq2seq.LuongAttention(num_units=self.enc_dim, memory=tiled_encoder_outputs)
    attention_cell = seq2seq.AttentionWrapper(dec_cell, attention_mechanism, self.enc_dim//2)

    dec_init_state = attention_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size*beam_width)
    if True:
      decoder = seq2seq.BasicDecoder(
        cell=attention_cell,
        helper=helper,
        initial_state=dec_init_state,
        output_layer=output_layer)
    else:
      decoder = seq2seq.BeamSearchDecoder(
        cell=attention_cell,
        embedding=W_E,
        start_tokens=tf.constant(1, dtype=tf.int32, shape=[self.batch_size]),
        end_token=2,
        beam_width=beam_width,
        initial_state=dec_init_state,
        output_layer=output_layer
      )
    outputs, state, seq_len = seq2seq.dynamic_decode(decoder)

    if True:
      logit = tf.reshape(outputs.rnn_output, [self.batch_size, self.max_sent_len, self.vocab_size])
      return logit[:, 1:, :]
    else:
      return tf.expand_dims(outputs.predicted_ids[:, :, 0], 2)