from agent_dir.agent import Agent
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        self.args = args
        super(Agent_DQN, self).__init__(env)
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        graph = tf.Graph()
        with graph.as_default():
            initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
            with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
                self.anime_holder = tf.placeholder(tf.float32, [None, 84, 84, 4])
                self.action_holder = tf.placeholder(tf.float32, [None, 4])
                self.Q_pred = self.build_model()
                scope.reuse_variables()

            # self.sess = tf.Session(graph=graph)
            # self.sess.run(tf.global_variables_initializer())

            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.config.graph_options.optimizer_options.global_jit_level = \
                tf.OptimizerOptions.ON_1
            self.sv = tf.train.Supervisor(logdir=self.args.log_dir,
                                     save_model_secs=self.args.save_model_secs)




    def build_model(self):
        conv1 = tf.layers.conv2d(self.anime_holder, filters=32, kernel_size=5,
                                padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=5,
                                padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=5,
                                padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)


        flatten = tf.reshape(pool3, [-1, 64*(10**2)])
        concat = tf.concat([flatten, self.action_holder], axis=1)

        dense1 = tf.layers.dense(concat, units=512, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, units=256, activation=tf.nn.relu)
        return tf.layers.dense(dense2, units=1)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """

        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        rewards = []
        for n_episode in range(self.args.max_ep):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            episode_reward = 0.0

            # playing one game
            while (not done):
                action = self.make_action(state, test=False)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            print(episode_reward)

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
        action_batch = np.diag([1]*4)
        anime_batch = np.repeat(np.expand_dims(observation, axis=0), repeats=4, axis=0)

        with self.sv.managed_session(config=self.config) as sess:
            Q_vals = sess.run(self.Q_pred, feed_dict={self.anime_holder: anime_batch,
                                         self.action_holder: action_batch})

            print(np.max(Q_vals)-np.min(Q_vals))
            return np.argmax(Q_vals)

