import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import random
import tensorflow as tf
import time
import pickle
from collections import deque
from matplotlib import pyplot as plt


def prepro(o, image_size=(80, 80)):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32), axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG, self).__init__(env)
        self.args = args
        self.rewards = []
        self.n_episode = 0
        self.action_space = 3
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.optimizer_options.global_jit_level = \
            tf.OptimizerOptions.ON_1
        self.ep_obs = []
        self.ep_actions = []
        self.ep_rewards = []

        if args.test_pg:
            #you can load your model here
            print('loading trained model')


        self.sess = tf.Session(config=self.config)
        model_path = os.path.join(self.args.log_dir, 'model.meta')
        if os.path.isfile(model_path):
            print('Loading saved model.')
            self.saver = tf.train.import_meta_graph(model_path)
            self.saver.restore(self.sess, os.path.join(self.args.log_dir, 'model'))
            self.obs_holder = tf.get_collection("obs_holder")[0]
            self.actions_holder = tf.get_collection("actions_holder")[0]
            self.val_holder = tf.get_collection("val_holder")[0]
            self.act_probs = tf.get_collection("act_probs")[0]
            self.train_op = tf.get_collection("train_op")[0]
        else:
            print('Creating new model.')
            self.build_model()
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

        log_path = os.path.join(self.args.log_dir, 'myLogs')
        if os.path.isfile(log_path):
            with open(log_path, 'rb') as file:
                logs = pickle.load(file)
                self.n_episode = logs['n_episode']
                self.rewards = logs['rewards']

    def build_model(self):
        self.obs_holder = tf.placeholder(tf.float32, [None, 80, 80, 1], name="observations")
        self.actions_holder = tf.placeholder(tf.int32, [None, ], name="actions")
        self.val_holder = tf.placeholder(tf.float32, [None, ], name="values")

        conv1 = tf.layers.conv2d(self.obs_holder, padding='same', filters=16, kernel_size=8, strides=4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, filters=32, padding='same', kernel_size=4, strides=2, activation=tf.nn.relu)
        flatten = tf.reshape(conv2, [-1, 32*(10**2)])
        dense1 = tf.layers.dense(flatten, 128, activation=tf.nn.relu)
        self.act_probs = tf.layers.dense(dense1, self.action_space, activation=tf.nn.softmax, name='act_probs')  # use softmax to convert to probability

        # log_prob = tf.reduce_sum(tf.log(self.act_probs)*tf.one_hot(self.actions_holder, self.action_space), axis=1)
        # self.loss = tf.reduce_sum(-log_prob*self.val_holder, name='loss')  # reward guided loss
        # self.train_op = tf.train.RMSPropOptimizer(self.args.learning_rate, self.args.lr_decay_rate).minimize(self.loss)

        tf_y = tf.one_hot(self.actions_holder, self.action_space)
        tf_discounted_epr = tf.expand_dims(self.val_holder, 1)
        loss = tf.nn.l2_loss(tf_y - self.act_probs)
        optimizer = tf.train.RMSPropOptimizer(self.args.learning_rate, decay=self.args.lr_decay_rate)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
        self.train_op = optimizer.apply_gradients(tf_grads)


        tf.add_to_collection("obs_holder", self.obs_holder)
        tf.add_to_collection("actions_holder", self.actions_holder)
        tf.add_to_collection("val_holder", self.val_holder)
        tf.add_to_collection("act_probs", self.act_probs)
        tf.add_to_collection("train_op", self.train_op)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.ep_obs = []
        self.ep_actions = []
        self.ep_rewards = []
        self.pre_state = None

    def clean(self):
        if not self.args.test_pg:
            log_path = os.path.join(self.args.log_dir, 'myLogs')
            with open(log_path, 'wb+') as file:
                logs = {}
                logs['n_episode'] = self.n_episode
                logs['rewards'] = self.rewards
                pickle.dump(logs, file)
        self.sess.close()
        print('program stopped.')

    def train(self):
        """
        Implement your training algorithm here
        """
        while self.n_episode < self.args.max_episode:
            self.init_game_setting()
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            # playing one game
            while (not done):
                if self.n_episode % 50 == 0:
                    self.env.env.render()
                action = self.make_action(state, test=False)
                state, reward, done, info = self.env.step(action) # Notice
                self.store_exp(action-1, reward)
                episode_reward += reward

            self.rewards.append(episode_reward)
            prepro_rewards = self.prepro_reward()
            self.sess.run(self.train_op, feed_dict={
                self.obs_holder: np.array(self.ep_obs),
                self.actions_holder: np.array(self.ep_actions, dtype=int),
                self.val_holder: prepro_rewards})
            if self.n_episode % 10 == 0:
                print('n_episode:', self.n_episode, end=' ')
                print('avg_score:', np.mean(self.rewards[-10:]))
            else:
                print('\tn_episode:', self.n_episode, end=' ')
                print('score:', episode_reward)

            if self.n_episode % 100 == 0:
                self.saver.save(self.sess, os.path.join(self.args.log_dir, 'model'))

            self.n_episode += 1

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        if self.pre_state is None:
            prepro_state = np.zeros([80, 80, 1], dtype=float)
        else:
            prepro_state = prepro(observation) - self.pre_state
        probs = self.sess.run(self.act_probs,
                              feed_dict={self.obs_holder: np.expand_dims(prepro_state, 0)})

        action = np.random.choice(self.action_space, p=probs[0])
        # plt.imshow(prepro_state[:, :, 0], cmap='gray')
        # plt.show()
        self.pre_state = prepro(observation)
        self.obs = prepro_state # to be pushed in memory
        return action+1

    def prepro_reward(self):
        # discount and normalize
        discount_rewards = np.zeros([len(self.ep_rewards)])
        running_sum = 0
        for i in range(len(self.ep_rewards)-1, -1, -1):
            running_sum = running_sum * self.args.GAMMA + self.ep_rewards[i]
            discount_rewards[i] = running_sum

        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards + 1e-3)
        return discount_rewards

    def store_exp(self, action, reward):
        self.ep_obs.append(self.obs)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)