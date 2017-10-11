import numpy as np
from tensorflow.contrib import keras
from tqdm import tqdm
import tensorflow as tf
import argparse
import random
import copy

n_epoch = 10000
window_size = 128
batch_size = 512
n_class = 48
dim = 69
lr = 0.001

def get_intchar_map(path):
	phone2int = {}
	phone2char = {}
	with open(path) as file:
		for line in file:
			sep = line[:-1].split('\t')
			phone2int[sep[0]] = int(sep[1])
			phone2char[sep[0]] = str(sep[2])
	return phone2int, phone2char
phone2int, phone2char = get_intchar_map('./data/48phone_char.map')
int2phone = {v: k for k, v in phone2int.items()}

def get_phone_map(path):
	phone_map = {}
	with open(path) as file:
		for line in file:
			sep = line[:-1].split('\t')
			phone_map[sep[0]] = str(sep[1])
	return phone_map
phone_map = get_phone_map('./data/48_39.map')

def get_phone(pred):
	assert(pred.ndim == 1)
	phones = []
	for i in pred:
		phones.append(phone_map[int2phone[i]])
	return phones

def load_data(mode = 'train'):
	path = ''
	if mode == 'train':
		path = './data/trainframes.npy'
		file = open('./data/labels.npy','rb')
		labels = np.load(file)
		file.close()
	elif mode == 'test':
		path = './data/testframes.npy'
		labels = None

	file = open(path,'rb')
	frames = np.load(file)
	file.close()

	print('Data loaded.')
	return frames, labels

def get_batch(frames,labels):
	def get_single_ex(frames, labels):
		start_id = random.randint(0,len(frames)-window_size-1)
		ex_frames = frames[start_id:start_id + window_size]
		ex_labels = labels[start_id:start_id + window_size]
		return ex_frames, ex_labels
	batch_frames = np.zeros([batch_size, window_size, dim], dtype=float)
	batch_labels = np.zeros([batch_size, window_size, n_class], dtype=float)
	for idx in range(batch_size):
		ex_frames, ex_labels = get_single_ex(frames, labels)
		batch_frames[idx] = ex_frames.copy()
		batch_labels[idx] = keras.utils.to_categorical(ex_labels, n_class).copy()
	return batch_frames, batch_labels

def model(feed_frames):
	gru_units = 64
	cell = tf.contrib.rnn.GRUCell(num_units=gru_units)
	init_state = cell.zero_state(batch_size, dtype=tf.float32)
	outputs, _ = tf.nn.dynamic_rnn(cell, feed_frames, initial_state=init_state, time_major=False)
	flatten_outputs = tf.reshape(outputs,[-1, gru_units])
	dense1 = tf.layers.dense(flatten_outputs, 512, activation=tf.nn.relu)
	dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu)
	dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu)
	pred = tf.layers.dense(dense3, n_class)
	return pred

frames, labels = load_data('train')
cat_labels = keras.utils.to_categorical(labels, n_class)

feed_frames = tf.placeholder(tf.float32,[None, window_size, dim])
feed_labels = tf.placeholder(tf.float32,[None, window_size, n_class])
flatten_labels = tf.reshape(feed_labels, [-1, n_class])
pred = model(feed_frames)

loss = tf.losses.softmax_cross_entropy(onehot_labels=flatten_labels, logits=pred)
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(flatten_labels,1),tf.arg_max(pred,1)), tf.float32))

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(n_epoch):
		batch_frames, batch_labels = get_batch(frames, labels)
		sess.run(train_op, feed_dict={feed_frames: batch_frames,
									  feed_labels: batch_labels})

		if (i + 1) % 10 == 0:
			print(i, sess.run([loss, accuracy], feed_dict={feed_frames: batch_frames,
														   feed_labels: batch_labels}))

test_frames, _ = load_data('test')
print(test_frames.shape)