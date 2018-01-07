import argparse
import tensorflow as tf
from utils import*

class GAN:
    def __init__(self, args):
        self.args = args

        self.isTrain_holder = tf.placeholder(tf.bool)
        self.feat_holder = tf.placeholder(tf.float32, [None, 2*args.emb_dim])
        with tf.variable_scope('gen'):
            self.noise_holder, self.fake_img = self.generator()

        with tf.variable_scope('dis'):
            self.img_holder = tf.placeholder(tf.float32, [None, 64, 64, 3])
            self.d_noise_holder = tf.placeholder(tf.float32, [1])
            self.random_feat_holder = tf.placeholder(tf.float32, [None, 2*args.emb_dim])
            self.rr_logits =\
                self.discriminator(self.img_holder, self.feat_holder, None)

            self.fr_logits =\
                self.discriminator(self.fake_img, self.feat_holder, True)

            self.rf_logits =\
                self.discriminator(self.img_holder, self.random_feat_holder, True)

            self.ff_logits =\
                self.discriminator(self.fake_img, self.random_feat_holder, True)

        self.gen_vars = []
        self.dis_vars = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            if "gen" in var.name:
                self.gen_vars.append(var)
            elif "dis" in var.name:
                self.dis_vars.append(var)
        for var in self.gen_vars:
            print(var)
        print('')
        for var in self.dis_vars:
            print(var)

        self.D_loss, self.G_loss, self.opt_dis, self.opt_gen = self.get_losses()



    def get_losses(self):
        def cross_entropy(label, logit):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
        rr_loss = cross_entropy(tf.constant(1., tf.float32, [self.args.batch_size, 1]), self.rr_logits)
        rf_loss = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.rf_logits)
        fr_loss = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.fr_logits)
        ff_loss = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.ff_logits)
        D_loss = rr_loss + rf_loss + fr_loss + ff_loss

        G_loss_1 = cross_entropy(tf.constant(1., tf.float32, [self.args.batch_size, 1]), self.fr_logits)
        G_loss_2 = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.ff_logits)
        G_loss = G_loss_1 + G_loss_2

        dis_optimizer = tf.train.AdamOptimizer(self.args.lr, 0.5)
        opt_dis = dis_optimizer.minimize(D_loss, var_list=self.dis_vars)

        gen_optimizer = tf.train.AdamOptimizer(self.args.lr, 0.5)
        opt_gen = gen_optimizer.minimize(G_loss, var_list=self.gen_vars)

        return D_loss, G_loss, opt_dis, opt_gen

    def generator(self):
        isTrain = self.isTrain_holder
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat = tf.concat([noise_holder, self.feat_holder], axis=1)

        dense = tf.layers.dense(concat, 128*(4**2))
        reshape = tf.reshape(dense, [-1, 4, 4, 128])
        lrelu0 = lrelu(tf.layers.batch_normalization(reshape, training=isTrain))

        deconv2 = tf.layers.conv2d_transpose(lrelu0, 128, 5, (2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain))

        deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, 5, (2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain))

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 64, 5, (2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(deconv4, training=isTrain))

        deconv5 = tf.layers.conv2d_transpose(lrelu4,  3, 5, (2, 2), padding='same', activation=tf.nn.tanh)

        return noise_holder, deconv5

    def discriminator(self, input, feats, reuse):
        isTrain = self.isTrain_holder
        if reuse:
            tf.get_variable_scope().reuse_variables()

        input += tf.random_normal([self.args.batch_size, 64, 64, 3],
                                  stddev=0.1)

        conv1 = tf.layers.conv2d(input, 32, 5, 2, padding='same', name='conv1')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain, name='batch0'))

        conv2 = tf.layers.conv2d(lrelu1, 64, 5, 2, padding='same', name='conv2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain, name='batch1'))

        conv3 = tf.layers.conv2d(lrelu2, 128, 5, 2, padding='same', name='conv3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain, name='batch2'))

        conv4 = tf.layers.conv2d(lrelu3, 128, 5, 2, padding='same', name='conv4')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain, name='batch3'))

        flat = tf.reshape(lrelu4, [-1, 128*(4**2)])
        concat = tf.concat([flat, feats], axis=1)
        dense1 = tf.layers.dense(concat, 512, activation=lrelu, name='dense1')
        logits = tf.layers.dense(dense1, 1, name='o')
        return logits

class WGAN(GAN):
    def get_losses(self):
        pass


