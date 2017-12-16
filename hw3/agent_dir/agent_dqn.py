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

'''
Dueling DQN Implementation
'''

class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN, self).__init__(env)
        self.args = args
        if self.args.test_dqn:
            self.args.log_dir = './logs_collection/dqn_logs_lrelu2'
        print(vars(args))
        self.action_space = env.action_space.n
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.optimizer_options.global_jit_level = \
            tf.OptimizerOptions.ON_1
        self.env_step = 0
        self.rewards = []
        self.sess = tf.Session(config=self.config)

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
            if self.args.memType == 'prioritized':
                self.abs_error = tf.get_collection("abs_error", scope=scope)[0]
                self.ISWeights_holder = tf.get_collection("ISWeights_holder", scope=scope)[0]

            scope = "target_network"
            self.tar_anime = tf.get_collection("anime_holder", scope=scope)[0]
            self.tar_action = tf.get_collection("action_holder", scope=scope)[0]
            self.tar_Q_pred =tf.get_collection("Q_pred", scope=scope)[0]
        else:
            print('Creating new model.')
            with tf.variable_scope('main_network'):
                self.main_anime, self.main_action, self.main_Q_pred = self.build_model()
                self.main_Q_tar = tf.placeholder(tf.float32, [None], name='Q_tar')
                if self.args.memType == 'basic':
                    self.loss = tf.losses.mean_squared_error(self.main_Q_tar, self.main_Q_pred)
                elif self.args.memType == 'prioritized':
                    self.ISWeights_holder = tf.placeholder(tf.float32, [None], name='ISWeights')
                    self.loss = tf.reduce_mean(self.ISWeights_holder*tf.squared_difference(self.main_Q_tar, self.main_Q_pred))
                    tf.add_to_collection("ISWeights_holder", self.ISWeights_holder)

                self.abs_error = tf.abs(self.main_Q_tar - self.main_Q_pred)
                self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate, decay=self.args.lr_decay_rate).minimize(self.loss)

                tf.add_to_collection("Q_tar", self.main_Q_tar)
                tf.add_to_collection("loss", self.loss)
                tf.add_to_collection("optimize", self.optimize)
                tf.add_to_collection("abs_error", self.abs_error)

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

        if self.args.memType == 'basic':
            self.replayMem = replayMem(args.memSize, mode='basic')
        elif self.args.memType == 'prioritized':
            self.replayMem = replayMem(args.memSize, mode='prioritized')
        if not self.args.test_dqn:
            if os.path.isfile(self.args.memLog):
                self.replayMem.load(self.args.memLog)
            else:
                print('Creating new replayMem')
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
            betas = tf.get_variable('beta', x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)

            return x*tf.nn.sigmoid(betas*x)

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
        if self.args.memType == 'basic':
            batch = self.replayMem.sample(batch_size=self.args.batch_size)
        elif self.args.memType == 'prioritized':
            self.tree_idx, batch, self.ISWeights = self.replayMem.sample(batch_size=self.args.batch_size)

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
        if self.args.double:
            all_Q = self.sess.run(self.main_Q_pred, {self.main_anime: full_state,
                                                     self.main_action: all_actions})
            max_Q_id = np.argmax(all_Q.reshape([self.args.batch_size, self.action_space]), 1)

            ## Convert to one-hot
            one_hot = np.zeros([self.args.batch_size, self.action_space])
            for i in range(self.args.batch_size):
                idx = max_Q_id[i]
                one_hot[i][idx] = 1

            max_Q = self.sess.run(self.tar_Q_pred, {self.tar_anime: batch_nextstates,
                                                    self.tar_action: one_hot})
        else:
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

                        if self.args.memType == 'basic':
                            self.sess.run(self.optimize, feed_dict=feed_dict)
                        elif self.args.memType == 'prioritized':
                            feed_dict[self.ISWeights_holder] = self.ISWeights
                            _, errors = self.sess.run([self.optimize, self.abs_error], feed_dict=feed_dict)
                            self.replayMem.batch_update(self.tree_idx, errors)

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
    epsilon = 0.01
    alpha = 0.7
    beta = 0.5
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, max_size, mode='basic'):
        self.mode = mode
        self.max_size = max_size
        if self.mode == 'basic':
            self.container = deque(maxlen=max_size)
        elif self.mode == 'prioritized':
            self.container = SumTree(max_size)

    def sample(self, batch_size):
        if self.mode == 'basic':
            return random.sample(self.container, batch_size)
        elif self.mode == 'prioritized':
            b_idx, b_memory, ISWeights = \
                np.empty((batch_size,), dtype=np.int32), \
                [], \
                np.empty((batch_size))

            pri_seg = self.container.total_p / batch_size  # priority segment
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

            min_prob = np.min(self.container.tree[-self.container.capacity:]) / self.container.total_p  # for later calculate ISweight
            for i in range(batch_size):
                a, b = pri_seg * i, pri_seg * (i + 1)
                v = np.random.uniform(a, b)
                idx, p, data = self.container.get_leaf(v)
                prob = p / self.container.total_p
                ISWeights[i] = np.power(prob / min_prob, -self.beta)
                b_idx[i] = idx
                b_memory.append(data)
            return b_idx, b_memory, ISWeights

    def push(self, newDat):
        if self.mode == 'basic':
            self.container.append(newDat)
        elif self.mode == 'prioritized':
            max_p = np.max(self.container.tree[-self.max_size:])
            if max_p == 0:
                max_p = self.abs_err_upper
            self.container.add(max_p, newDat)

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.container.update(ti, p)

    # def set_mode(self, mode):
    #     self.mode = mode

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


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root
