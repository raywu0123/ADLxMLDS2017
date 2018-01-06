import tensorflow as tf
import numpy as np
import random
from argument import parse_args
from model import GAN
import os
import scipy.misc as misc
from utils import*
from skimage import transform


args = parse_args()
tags = get_tags(args.data_dir)
# glove = loadGloveModel('./glove.6B/glove.6B.'+str(args.emb_dim)+'d.txt')

dir_list = []
for filename in os.listdir(os.path.join(args.data_dir, 'faces')):
    # tag = tags[filename.strip('.jpg')]
    # if not (len(tag[0]) == 0 and len(tag[1]) == 0):
    dir_list.append(filename)
print('Total ', len(dir_list), ' images in training data.')

faces = {}
for filename in dir_list:
    filepath = os.path.join(os.path.join(args.data_dir, 'faces/'+filename))
    img = misc.imread(filepath)
    img = transform.resize(img, (64, 64))
    faces[filename] = img*2 - 1


def get_batch():
    batch_feat = np.zeros([args.batch_size, args.emb_dim*2], dtype=float)
    batch_img = np.zeros([args.batch_size, 64, 64, 3], dtype=float)
    for i in range(args.batch_size):
        filename = random.choice(dir_list)
        # batch_feat[i] = glove_feats(glove, tags[filename.strip('.jpg')], args)
        if filename.strip('.jpg') in tags:
            batch_feat[i] = one_hot_feats(tags[filename.strip('.jpg')], args)
        else:
            batch_feat[i] = one_hot_feats([[], []], args)
        batch_img[i] = faces[filename]
        # print(tags[filename.strip('.jpg')])
        # print(batch_feat[i][:args.emb_dim])
        # print(batch_feat[i][args.emb_dim:])
        # print(batch_img[i])

    return batch_img, batch_feat



if __name__ == '__main__':
    if not os.path.exists(args.save_img_dir):
        os.mkdir(args.save_img_dir)
    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
            model = GAN(args)
            scope.reuse_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level =\
        tf.OptimizerOptions.ON_1
        sv = tf.train.Supervisor(logdir=args.log_dir,
                             save_model_secs=args.save_model_secs)

        with sv.managed_session(config=config) as sess:
            for n_epoch in range(args.max_epoch):
                batch_noise = np.random.standard_normal([args.batch_size, args.noise_dim])
                batch_img, batch_feat = get_batch()
                batch_random_feat = arti_feats(args)
                _, D_loss_curr = sess.run([model.opt_dis, model.D_loss],
                                 feed_dict={model.isTrain_holder: True,
                                            model.noise_holder: batch_noise,
                                            model.feat_holder: batch_feat,
                                            model.img_holder: batch_img,
                                            model.random_feat_holder: batch_random_feat
                                            })

                _, G_loss_curr = sess.run([model.opt_gen, model.G_loss],
                                  feed_dict={model.isTrain_holder: True,
                                             model.noise_holder: batch_noise,
                                             model.feat_holder: batch_feat,
                                             model.random_feat_holder: batch_random_feat
                                             })

                if n_epoch % args.info_epoch == 0:
                    print('n_epoch: ', n_epoch)
                    print("\tGenerator Loss: ", G_loss_curr)
                    print("\tDiscirminator Loss: ", D_loss_curr)
                    batch_noise = np.random.standard_normal([args.batch_size, args.noise_dim])
                    batch_feat = arti_feats(args, [['blue'], ['red']])
                    generated_images = sess.run(model.fake_img,
                                      feed_dict={model.isTrain_holder: False,
                                                 model.noise_holder: batch_noise,
                                                 model.feat_holder: batch_feat
                                                 })
                    # plot(generated_images[:16, :, :, 1]/2 + 0.5, 'epoch_' + str(n_epoch))
                    filename = 'epoch_' + str(n_epoch) + '.jpg'
                    misc.imsave(os.path.join(args.save_img_dir, filename), generated_images[0, :, :, :]/2 + 0.5)