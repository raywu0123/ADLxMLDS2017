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

        self.D_loss, self.G_loss, self.opt_dis, self.opt_gen = self.get_losses()

    def get_losses(self):
        def cross_entropy(label, logit):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

        rr_loss = cross_entropy(tf.constant(1., tf.float32, [self.args.batch_size, 1]), self.rr_logits)
        rf_loss = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.rf_logits)
        fr_loss = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.fr_logits)
        ff_loss = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.ff_logits)
        D_loss = rr_loss + (rf_loss + fr_loss + ff_loss)

        G_loss_1 = cross_entropy(tf.constant(1., tf.float32, [self.args.batch_size, 1]), self.fr_logits)
        G_loss_2 = cross_entropy(tf.constant(0., tf.float32, [self.args.batch_size, 1]), self.ff_logits)
        G_loss = G_loss_1 #+ G_loss_2

        self.gen_vars, self.dis_vars = self.get_vars()
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

        deconv1 = batch_deconv(lrelu0, 128, 5, (2, 2), 'same', '1', isTrain)
        deconv2 = batch_deconv(deconv1, 128, 5, (2, 2), 'same', '2', isTrain)
        deconv3 = batch_deconv(deconv2, 64, 5, (2, 2), 'same', '3', isTrain)
        deconv4 = batch_deconv(deconv3, 32, 5, (2, 2), 'same', '4', isTrain)
        deconv5 = tf.layers.conv2d_transpose(deconv4,  3, 5, (1, 1), padding='same', activation=tf.nn.tanh)

        return noise_holder, deconv5

    def discriminator(self, input, feats, reuse):
        isTrain = self.isTrain_holder
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv1 = batch_conv(input, 32, 5, 2, 'same', '1', isTrain)
        conv2 = batch_conv(conv1, 64, 5, 2, 'same', '2', isTrain)
        conv3 = batch_conv(conv2, 128, 5, 2, 'same', '3', isTrain)
        conv4 = batch_conv(conv3, 128, 5, 2, 'same', '4', isTrain)

        flat = tf.reshape(conv4, [-1, 128*(4**2)])
        concat = tf.concat([flat, feats], axis=1)
        dense1 = tf.layers.dense(concat, 512, activation=lrelu, name='dense1')
        logits = tf.layers.dense(dense1, 1, name='o')
        return logits

    def get_vars(self):
        gen_vars = []
        dis_vars = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            if "gen" in var.name:
                gen_vars.append(var)
            elif "dis" in var.name:
                dis_vars.append(var)
        for var in gen_vars:
            print(var)
        print('')
        for var in dis_vars:
            print(var)
        return gen_vars, dis_vars

class WGAN(GAN):
    def get_losses(self):
        D_loss = tf.reduce_mean(self.rr_logits - (self.fr_logits + self.rf_logits + self.ff_logits)/3)
        G_loss = tf.reduce_mean(self.fr_logits)

        dis_optimizer = tf.train.RMSPropOptimizer(self.args.lr)
        opt_dis = dis_optimizer.minimize(D_loss, var_list=self.dis_vars)

        gen_optimizer = tf.train.RMSPropOptimizer(self.args.lr)
        opt_gen = gen_optimizer.minimize(G_loss, var_list=self.gen_vars)

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.dis_vars]
        return D_loss, G_loss, opt_dis, opt_gen

class WGAN_GP(GAN):
    def get_losses(self):
        with tf.variable_scope('dis'):
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon*self.img_holder + (1 - epsilon)*self.fake_img
            d_hat = self.discriminator(x_hat, self.feat_holder, True)
            ddx = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=[1, 2, 3]))
            ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.args.gp_scale)
            D_loss = tf.reduce_mean(self.rr_logits - (self.fr_logits + self.rf_logits + self.ff_logits)/3)
            D_loss += ddx

        with tf.variable_scope('gen'):
            G_loss = tf.reduce_mean(self.fr_logits)


        self.gen_vars, self.dis_vars = self.get_vars()
        with tf.variable_scope('dis'):
            dis_optimizer = tf.train.AdamOptimizer(self.args.lr, 0.5)
            opt_dis = dis_optimizer.minimize(D_loss, var_list=self.dis_vars)

        with tf.variable_scope('gen'):
            gen_optimizer = tf.train.AdamOptimizer(self.args.lr, 0.5)
            opt_gen = gen_optimizer.minimize(G_loss, var_list=self.gen_vars)

        return D_loss, G_loss, opt_dis, opt_gen

    def generator(self):
        isTrain = self.isTrain_holder
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat = tf.concat([noise_holder, self.feat_holder], axis=1)

        dense = tf.layers.dense(concat, 128*(4**2))
        reshape = tf.reshape(dense, [-1, 4, 4, 128])
        lrelu0 = lrelu(reshape)

        deconv1 = deconv(lrelu0, 128, 5, (2, 2), 'same', '1')
        deconv2 = deconv(deconv1, 128, 5, (2, 2), 'same', '2')
        deconv3 = deconv(deconv2, 64, 5, (2, 2), 'same', '3')
        deconv4 = deconv(deconv3, 32, 5, (2, 2), 'same', '4')
        deconv5 = tf.layers.conv2d_transpose(deconv4,  3, 5, (1, 1), padding='same', activation=tf.nn.tanh)

        return noise_holder, deconv5

    def discriminator(self, input, feats, reuse):
        isTrain = self.isTrain_holder
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv1 = conv(input, 32, 5, 2, 'same', '1')
        conv2 = conv(conv1, 64, 5, 2, 'same', '2')
        conv3 = conv(conv2, 128, 5, 2, 'same', '3')
        conv4 = conv(conv3, 128, 5, 2, 'same', '4')

        flat = tf.reshape(conv4, [-1, 128*(4**2)])
        concat = tf.concat([flat, feats], axis=1)
        dense1 = tf.layers.dense(concat, 512, activation=lrelu, name='dense1')
        logits = tf.layers.dense(dense1, 1, name='o')
        return logits


class ACGAN(GAN):
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

            self.Arr_logits, self.Dr_logit = self.discriminator(self.img_holder, self.feat_holder, None)
            self.Afr_logits, self.Df_logit = self.discriminator(self.fake_img, self.feat_holder, True)
            self.Arf_logits, _ = self.discriminator(self.img_holder, self.random_feat_holder, True)
            self.Aff_logits, _ = self.discriminator(self.fake_img, self.random_feat_holder, True)

        self.D_loss, self.G_loss, self.opt_dis, self.opt_gen = self.get_losses()

    def discriminator(self, input, feats, reuse):
        isTrain = self.isTrain_holder
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv1 = conv(input, 32, 5, 2, 'same', '1')
        conv2 = conv(conv1, 64, 5, 2, 'same', '2')
        conv3 = conv(conv2, 128, 5, 2, 'same', '3')
        conv4 = conv(conv3, 128, 5, 2, 'same', '4')

        flat = tf.reshape(conv4, [-1, 128*(4**2)])
        dense2 = tf.layers.dense(flat, 512, activation=lrelu, name='dense2')
        D_logits = tf.layers.dense(dense2, 1, name='D_logits')

        concat = tf.concat([flat, feats], axis=1)
        dense1 = tf.layers.dense(concat, 512, activation=lrelu, name='dense1')
        A_logits = tf.layers.dense(dense1, 1, name='A_logits')
        return A_logits, D_logits

    def generator(self):
        isTrain = self.isTrain_holder
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat = tf.concat([noise_holder, self.feat_holder], axis=1)

        dense = tf.layers.dense(concat, 128*(4**2))
        reshape = tf.reshape(dense, [-1, 4, 4, 128])
        lrelu0 = lrelu(reshape)

        deconv1 = batch_deconv(lrelu0, 128, 5, (2, 2), 'same', '1', isTrain)
        deconv2 = batch_deconv(deconv1, 128, 5, (2, 2), 'same', '2', isTrain)
        deconv3 = batch_deconv(deconv2, 64, 5, (2, 2), 'same', '3', isTrain)
        deconv4 = batch_deconv(deconv3, 32, 5, (2, 2), 'same', '4', isTrain)
        deconv5 = tf.layers.conv2d_transpose(deconv4,  3, 5, (1, 1), padding='same', activation=tf.nn.tanh)

        return noise_holder, deconv5

    def get_losses(self):
        def cross_entropy(logit, label):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
        def const(val):
            return tf.constant(val, tf.float32, [self.args.batch_size, 1])

        with tf.variable_scope('dis'):
            DD_Loss = cross_entropy(self.Dr_logit, const(1.0)) + cross_entropy(self.Df_logit, const(0.0))
            DA_Loss = cross_entropy(self.Arr_logits, const(1.0)) + cross_entropy(self.Arf_logits, const(0.0))\
                     + cross_entropy(self.Afr_logits, const(1.0)) + cross_entropy(self.Aff_logits, const(0.0))
            D_Loss = DD_Loss + DA_Loss
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * self.img_holder + (1 - epsilon) * self.fake_img
            d_hat, _ = self.discriminator(x_hat, self.feat_holder, True)
            ddx = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=[1, 2, 3]))
            ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.args.gp_scale)
            D_Loss += ddx

        with tf.variable_scope('gen'):
            GD_Loss = cross_entropy(self.Df_logit, const(1.0))
            GA_Loss = cross_entropy(self.Afr_logits, const(1.0)) + cross_entropy(self.Aff_logits, const(0.0))
            G_Loss = GD_Loss + GA_Loss

        self.gen_vars, self.dis_vars = self.get_vars()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            dis_optimizer = tf.train.AdamOptimizer(self.args.lr, 0.5)
            opt_dis = dis_optimizer.minimize(D_Loss, var_list=self.dis_vars)

            gen_optimizer = tf.train.AdamOptimizer(self.args.lr, 0.5)
            opt_gen = gen_optimizer.minimize(G_Loss, var_list=self.gen_vars)

        return D_Loss, G_Loss, opt_dis, opt_gen