import argparse
import tensorflow as tf


def lrelu(x, th=0.01):
    return tf.maximum(th * x, x)

class GAN:
    def __init__(self, args):
        self.args = args

        self.isTrain = tf.placeholder(tf.bool)
        self.feat_holder = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope('gen'):
            self.noise_holder, self.fake_img = self.generator()

        with tf.variable_scope('dis'):
            self.img_holder = tf.placeholder(tf.float32, [None, 28, 28, 1])
            self.random_feat_holder = tf.placeholder(tf.float32, [None, 10])
            self.rr_logit =\
                self.discriminator(self.img_holder, self.feat_holder, None)

            self.fr_logit =\
                self.discriminator(self.fake_img, self.feat_holder, True)

            self.rf_logit =\
                self.discriminator(self.img_holder, self.random_feat_holder, True)

            self.ff_logit =\
                self.discriminator(self.fake_img, self.random_feat_holder, True)


        with tf.variable_scope('dis'):
            self.D_loss = tf.reduce_mean(self.rr_logit - (self.fr_logit + self.ff_logit + self.rf_logit)/3)

            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon*self.img_holder + (1 - epsilon)*self.fake_img
            d_hat = self.discriminator(x_hat, self.feat_holder, True)
            ddx = tf.gradients(d_hat, x_hat)[0]
            print(ddx.get_shape().as_list())
            ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), [1, 2, 3]))
            ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.args.gp_scale)
            self.D_loss += ddx

        with tf.variable_scope('gen'):
            self.G_loss = tf.reduce_mean(self.fr_logit)

        G_vars = []
        D_vars = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            if "gen" in var.name:
                G_vars.append(var)
            elif "dis" in var.name:
                D_vars.append(var)
        for var in G_vars:
            print(var)
        print('')
        for var in D_vars:
            print(var)

        with tf.variable_scope('dis'):
            self.D_optimizer = tf.train.RMSPropOptimizer(self.args.lr)
            self.D_opt = self.D_optimizer.minimize(self.D_loss, var_list=D_vars)
            self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D_vars]

        with tf.variable_scope('gen'):
            self.G_optimizer = tf.train.RMSPropOptimizer(self.args.lr)
            self.G_opt = self.G_optimizer.minimize(self.G_loss, var_list=G_vars)

    def generator(self):
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat, 128, activation=tf.nn.relu)
        fake_img = tf.layers.dense(dense1, 784, activation=tf.nn.tanh)
        reshape3d = tf.reshape(fake_img, [-1, 28, 28, 1])
        return noise_holder, reshape3d

    def discriminator(self, input, feat, reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        reshape = tf.reshape(input, [-1, 784])
        concat = tf.concat([reshape, feat], axis=1)
        dense1 = tf.layers.dense(concat, 128, activation=tf.nn.relu, name='dense1')
        logits = tf.layers.dense(dense1, 1, activation=None, name='S_pred')
        return logits

class DCGAN(GAN):
    def generator(self):
        isTrain = self.isTrain
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat_feat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat_feat, 784)
        reshape_noise = tf.reshape(dense1, [-1, 7, 7, 16])
        lrelu0 = lrelu(tf.layers.batch_normalization(reshape_noise, training=isTrain))

        conv1 = tf.layers.conv2d_transpose(lrelu0, 32, [5, 5], strides=(2, 2), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 1, [5, 5], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv2)
        return noise_holder, o

    def discriminator(self, input, feat, reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        isTrain = self.isTrain
        # 1st hidden layer
        conv1 = tf.layers.conv2d(input, 32, 5, strides=(2, 2), padding='same', name='conv1')
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(conv1, 16, 5, strides=(2, 2), padding='same', name='conv2')

        flat = tf.reshape(conv2, [-1, 784])
        concat_feat = tf.concat([flat, feat], axis=1)
        dense2 = tf.layers.dense(concat_feat, 784, activation=lrelu, name='dense2')
        feat_logits = tf.layers.dense(dense2, 1, name='feat_logits')

        return feat_logits
