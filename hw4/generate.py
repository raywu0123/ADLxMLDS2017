import tensorflow as tf
import numpy as np
from argument import parse_args
from model import ACGAN as GAN
import os
import scipy.misc as misc
from utils import*

args = parse_args()
args.save_img_dir = './samples'
np.random.seed(0)

def parse_txt(line):
    line = line.strip('\n')
    line = line.split(',')
    id = line[0]
    condition = line[1]
    condition = condition.split(' ')
    pre_word = ''
    hair_feat = []
    eye_feat = []
    print(condition)
    for word in condition:
        if word == 'hair':
            hair_feat.append(pre_word)
        elif word == 'eyes':
            eye_feat.append(pre_word)
        pre_word = word

    return id, [hair_feat, eye_feat]

if __name__ == '__main__':
    if not os.path.exists(args.save_img_dir):
        os.mkdir(args.save_img_dir)
    with tf.Graph().as_default() as graph:
        with tf.variable_scope('model', reuse=None) as scope:
            model = GAN(args)
            scope.reuse_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level =\
        tf.OptimizerOptions.ON_1
        sv = tf.train.Supervisor(logdir=args.log_dir)
        with sv.managed_session(config=config) as sess:
            with open(args.test_txt, 'r') as txt:
                for line in txt:
                    batch_noise = np.random.standard_normal([args.batch_size, args.noise_dim])*0.7
                    id, feat = parse_txt(line)
                    batch_feat = arti_feats(args, feat)
                    generated_images = sess.run(model.fake_img,
                                      feed_dict={model.isTrain_holder: False,
                                                 model.noise_holder: batch_noise,
                                                 model.feat_holder: batch_feat
                                                 })
                    for i in range(5):
                        filename = 'sample_' + str(id) + '_' + str(i+1) + '.jpg'
                        misc.imsave(os.path.join(args.save_img_dir, filename), generated_images[i, :, :, :]/2 + 0.5)