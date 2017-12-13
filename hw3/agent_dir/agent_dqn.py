import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from agent_dir.agent import Agent
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from collections import deque
import random
import sys
import pickle

class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN, self).__init__(env)
        self.args = args
        self.replayMem = replayMem(args.memSize, mode='basic')

        self.action_space = env.action_space.n
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.optimizer_options.global_jit_level = \
            tf.OptimizerOptions.ON_1
        self.env_step = 0
        self.rewards = []
        self.sess = tf.Session(config=self.config)
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        model_path = os.path.join(self.args.log_dir, 'model.meta')
        if os.path.isfile(model_path):
            print('Loading saved model.')
            self.saver = tf.train.import_meta_graph(model_path)
            self.saver.restore(self.sess, os.path.join(self.args.log_dir, 'model'))

            scope = "main_network"
            self.main_anime = tf.get_collection("anime_holder", scope=scope)[0]
            self.main_action = tf.get_collection("action_holder", scope=scope)[0]
            self.main_Q_pred =tf.get_collection("Q_pred", scope=scope)[0]
            self.main_Q_tar =tf.get_collection("Q_tar", scope=scope)[0]
            self.loss = tf.get_collection("loss", scope=scope)[0]
            self.optimize = tf.get_collection("optimize", scope=scope)[0]

            scope = "target_network"
            self.tar_anime = tf.get_collection("anime_holder", scope=scope)[0]
            self.tar_action = tf.get_collection("action_holder", scope=scope)[0]
            self.tar_Q_pred =tf.get_collection("Q_pred", scope=scope)[0]
        else:
            print('Creating new model.')
            with tf.variable_scope('main_network'):
                self.main_anime, self.main_action, self.main_Q_pred = self.build_model()
                self.main_Q_tar = tf.placeholder(tf.float32, [None], name='Q_tar')
                self.loss = tf.losses.mean_squared_error(self.main_Q_tar, self.main_Q_pred)
                self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate, decay=self.args.lr_decay_rate).minimize(self.loss)
                tf.add_to_collection("Q_tar", self.main_Q_tar)
                tf.add_to_collection("loss", self.loss)
                tf.add_to_collection("optimize", self.optimize)

            with tf.variable_scope('target_network'):
                self.tar_anime, self.tar_action, self.tar_Q_pred = self.build_model()

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

        self.main_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_network')
        self.tar_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
        log_path = os.path.join(self.args.log_dir, 'myLogs')
        if os.path.isfile(log_path):
            with open(log_path, 'rb') as file:
                logs = pickle.load(file)
                self.env_step = logs['env_step']
                self.rewards = logs['rewards']

        if not self.args.test_dqn:
            self.replayMem.load(self.args.memLog)
            for var in self.main_vars:
                print(var)
            print('')
            for var in self.tar_vars:
                print(var)

    def clean(self):
        if not self.args.test_dqn:
            self.replayMem.save(self.args.memLog)
            log_path = os.path.join(self.args.log_dir, 'myLogs')
            with open(log_path, 'wb+') as file:
                logs = {}
                logs['env_step'] = self.env_step
                logs['rewards'] = self.rewards
                pickle.dump(logs, file)

        self.sess.close()
        print('program stopped.')

    def build_model(self):
        def swish(x):
            return x*tf.nn.sigmoid(x)

        def parametric_relu(_x):
            alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5

            return pos + neg

        def leaky_relu(x, alpha=0.01):
            return tf.maximum(x, alpha * x)

        anime_holder = tf.placeholder(tf.float32,
                                         [None, 84, 84, 4]
                                         ,name='anime_holder')
        action_holder = tf.placeholder(tf.float32,
                                          [None, self.action_space]
                                          ,name='action_holder')

        conv1 = tf.layers.conv2d(anime_holder, filters=32, kernel_size=8, strides=4,
                                padding='same', activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=4, strides=2,
                                padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, strides=1,
                                padding='same', activation=tf.nn.relu)


        flatten = tf.reshape(conv3, [-1, 64*(11**2)])
        concat = tf.concat([flatten, action_holder], axis=1)

        # dense1 = tf.layers.dense(concat, units=512, activation=tf.nn.relu)
        # dense1 = tf.layers.dense(concat, units=512, activation=parametric_relu)
        dense1 = tf.layers.dense(concat, units=512, activation=leaky_relu)
        Q_pred = tf.reshape(tf.layers.dense(dense1, units=1), [-1], name='Q_pred')


        tf.add_to_collection('anime_holder', anime_holder)
        tf.add_to_collection('action_holder', action_holder)
        tf.add_to_collection('Q_pred', Q_pred)
        return anime_holder, action_holder, Q_pred

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def get_batch(self):
        batch_states = np.zeros([self.args.batch_size, 84, 84, 4])
        batch_nextstates = np.zeros([self.args.batch_size, 84, 84, 4])
        batch_actions = np.zeros([self.args.batch_size, self.action_space])
        batch_rewards = np.zeros([self.args.batch_size])
        batch_done = np.zeros([self.args.batch_size], dtype=bool)
        batch = self.replayMem.sample(batch_size=self.args.batch_size)
        for i in range(self.args.batch_size):
            batch_states[i] = batch[i][0]
            batch_nextstates[i] = batch[i][1]
            # Convert to one-hot representation
            batch_actions[i][batch[i][2]] = 1
            batch_rewards[i] = batch[i][3]
            batch_done[i] = batch[i][4]

        nextstate_maxQ = self.get_maxQ(batch_nextstates)
        target_Q = batch_rewards + (~batch_done)*self.args.GAMMA * nextstate_maxQ

        return batch_states, batch_actions, target_Q

    def get_maxQ(self, batch_nextstates):
        full_state = np.repeat(batch_nextstates, self.action_space, 0)
        all_actions = np.tile(np.diag([1]*self.action_space), [self.args.batch_size, 1])
        all_Q = self.sess.run(self.tar_Q_pred, {self.tar_anime: full_state,
                                                self.tar_action: all_actions})
        max_Q = np.max(all_Q.reshape([self.args.batch_size, self.action_space]), 1)
        return max_Q

    def copy_network(self, main_col, target_col, verbose=False):
        for tar_var in target_col:
            for main_var in main_col:
                if main_var.name.strip('main_network/') == tar_var.name.strip('target_network/'):
                    self.sess.run(tar_var.assign(main_var))
                    if verbose:
                        print('Copying to ', tar_var.name)
                    break

    def train(self):
        """
        Implement your training algorithm here
        """
        while self.env_step < self.args.max_step:
            state = self.env.reset()
            self.init_game_setting()
            done = False
            episode_reward = 0.0
            # playing one game
            while (not done):
                action = self.make_action(state, test=False)
                next_state, reward, done, info = self.env.step(action)
                self.replayMem.push((state, next_state, action, reward, done))
                state = next_state
                episode_reward += reward
                if self.env_step % self.args.info_step == 0:
                    print('env_step:', self.env_step)
                if self.env_step > self.args.start_step:
                    if self.env_step % self.args.train_freq == 0:
                        batch_states, batch_actions, batch_targets = self.get_batch()
                        feed_dict = {self.main_anime: batch_states,
                                     self.main_action: batch_actions,
                                     self.main_Q_tar: batch_targets}

                        self.sess.run(self.optimize, feed_dict=feed_dict)
                    if self.env_step % self.args.tar_update_freq == 0:
                        self.copy_network(self.main_vars, self.tar_vars)
                        self.saver.save(self.sess, os.path.join(self.args.log_dir, 'model'))
                        print(' reward:', np.mean(self.rewards[-100:]))

                self.env_step += 1
            self.rewards.append(episode_reward)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        if test:
            explore_rate = 0.005
        else:
            if self.env_step > (self.args.max_step // 10):
                explore_rate = 0.05
            else:
                explore_rate = 1.0 - (1.0-0.05) * self.env_step / (self.args.max_step // 10)

        if random.uniform(0, 1) > explore_rate:
            action_batch = np.diag([1]*self.action_space)
            anime_batch = np.repeat(np.expand_dims(observation, axis=0), repeats=self.action_space, axis=0)
            Q_vals = self.sess.run(self.main_Q_pred,
                                   feed_dict={self.main_anime: anime_batch,
                                            self.main_action: action_batch})
            return np.argmax(Q_vals)
        else:
            return self.env.get_random_action()

class replayMem():
    def __init__(self, max_size, mode='basic'):
        self.mode = mode
        self.max_size = max_size
        if self.mode == 'basic':
            self.container = deque(maxlen=max_size)

    def sample(self, batch_size):
        return random.sample(self.container, batch_size)

    def push(self, newDat):
        if self.mode == 'basic':
            self.container.append(newDat)

    def set_mode(self, mode):
        self.mode = mode

    def is_full(self):
        return len(self.container) == self.container.maxlen

    def save(self, path):
        with open(path, 'wb+') as file:
            pickle.dump(self.container, file)
        print('MemLog saved.')

    def load(self, path):
        try:
            with open(path, 'rb') as file:
                self.container = pickle.load(file)
            print('Loaded replayMem from Log.')
        except:
            print('Creating new replayMem')

    @property
    def size(self):
        return len(self.container)
