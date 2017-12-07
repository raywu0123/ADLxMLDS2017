from agent_dir.agent import Agent
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from collections import deque
import random

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        self.args = args
        self.replayMem = replayMem(args.memSize, mode='basic')
        self.action_space = env.action_space.n
        super(Agent_DQN, self).__init__(env)
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        graph = tf.Graph()
        with graph.as_default():
            initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
            with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
                self.anime_holder = tf.placeholder(tf.float32, [None, 84, 84, 4])
                self.action_holder = tf.placeholder(tf.float32, [None, self.action_space])
                self.Q_holder = tf.placeholder(tf.float32, [None])
                self.Q_pred = self.build_model()
                scope.reuse_variables()

            self.loss = tf.losses.mean_squared_error(self.Q_holder, self.Q_pred)
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.config.graph_options.optimizer_options.global_jit_level = \
                tf.OptimizerOptions.ON_1
            self.sv = tf.train.Supervisor(logdir=self.args.log_dir,
                                     save_model_secs=self.args.save_model_secs)

    def build_model(self):
        conv1 = tf.layers.conv2d(self.anime_holder, filters=16, kernel_size=5,
                                padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=3,
                                padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        conv3 = tf.layers.conv2d(pool2, filters=32, kernel_size=3,
                                padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)


        flatten = tf.reshape(pool3, [-1, 32*(10**2)])
        concat = tf.concat([flatten, self.action_holder], axis=1)

        dense1 = tf.layers.dense(concat, units=128, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu)
        return tf.reshape(tf.layers.dense(dense2, units=1), [-1])

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
            batch_actions[i] = batch[i][2]
            batch_rewards[i] = batch[i][3]
            batch_done[i] = batch[i][4]

        nextstate_maxQ = self.get_maxQ(batch_nextstates)
        target_Q = batch_rewards + (~batch_done)*self.args.GAMMA * nextstate_maxQ

        return batch_states, batch_actions, target_Q

    def get_maxQ(self, batch_nextstates):
        full_state = np.repeat(batch_nextstates, self.action_space, 0)
        all_actions = np.tile(np.diag([1]*self.action_space), [self.args.batch_size,1])
        with self.sv.managed_session(config=self.config) as sess:
            all_Q = sess.run(self.Q_pred, {self.anime_holder: full_state,
                                           self.action_holder: all_actions})
            max_Q = np.argmax(all_Q.reshape([self.action_space, self.args.batch_size]), 0)

        return max_Q

    def train(self):
        """
        Implement your training algorithm here
        """
        rewards = []
        for n_episode in range(self.args.max_ep):
            print('n_ep:', n_episode, end=' ')
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

            rewards.append(episode_reward)
            print(' reward:', rewards[-1])
            if self.replayMem.size > self.args.batch_size:
                batch_states, batch_actions, batch_targets = self.get_batch()
                with self.sv.managed_session() as sess:
                    feed_dict = {self.anime_holder: batch_states,
                                 self.action_holder: batch_actions,
                                 self.Q_holder: batch_targets}
                    sess.run(self.optimize, feed_dict=feed_dict)
                    print('loss:', sess.run(self.loss, feed_dict=feed_dict))
            else:
                print('')
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
        if not test and random.uniform(0,1) > 1 - self.args.epsilon:
            action_batch = np.diag([1]*self.action_space)
            anime_batch = np.repeat(np.expand_dims(observation, axis=0), repeats=self.action_space, axis=0)
            with self.sv.managed_session(config=self.config) as sess:
                Q_vals = sess.run(self.Q_pred, feed_dict={self.anime_holder: anime_batch,
                                             self.action_holder: action_batch})
                return int(np.argmax(Q_vals))
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

    @property
    def size(self):
        return len(self.container)
