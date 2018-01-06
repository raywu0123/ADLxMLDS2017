import tensorflow as tf

def lrelu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(samples, filename):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)

    if not os.path.exists('./inference_imgs'):
        os.mkdir('./inference_imgs')
    plt.savefig('inference_imgs/{}.png'.format(filename.zfill(3)), bbox_inches='tight')
    plt.close(fig)

import numpy as np
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.", len(model), " words loaded!")
    return model


import os

hair_types = [  'orange hair', 'white hair', 'aqua hair', 'gray hair',
                'green hair', 'red hair', 'purple hair', 'pink hair',
                'blue hair', 'black hair', 'brown hair', 'blonde hair']

eye_types = [   'gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

color_list = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple',
              'pink', 'blue', 'black', 'brown', 'blonde', 'yellow']


color_map = { 'orange': 0, 'white': 1, 'aqua': 2, 'gray': 3,
               'green': 4, 'red': 5, 'purple': 6, 'pink': 7,
               'blue': 8, 'black': 9, 'brown': 10, 'blonde': 11, 'yellow': 11,
              'none': 12}



import random
def glove_feats(model, tag, args):
    hair_feat = random.choice(tag[0])
    eye_feat = random.choice(tag[1])
    full_feats = np.zeros([2, args.emb_dim], dtype=float)
    full_feats[0] = model[hair_feat]
    full_feats[1] = model[eye_feat]
    full_feats = full_feats.reshape([2*args.emb_dim])
    return full_feats

def one_hot_feats(tag, args):
    full_feats = np.zeros([2, args.emb_dim], dtype=float)
    if len(tag[0]) != 0:
        hair_feat = random.choice(tag[0])
    else:
        hair_feat = random.choice(color_list)

    if len(tag[1]) != 0:
        eye_feat = random.choice(tag[1])
    else:
        eye_feat = random.choice(color_list)

    full_feats[0, color_map[hair_feat]] = 1
    full_feats[1, color_map[eye_feat]] = 1
    full_feats = full_feats.reshape([2 * args.emb_dim])
    return full_feats



def get_tags(path):
    tags = {}
    with open(os.path.join(path, 'tags_clean.csv'), 'r') as file:
        for line in file:
            split_line = line.strip('\n').split(',')
            id = split_line[0]
            feats = split_line[1].split('\t')
            feats = [feat.split(':')[0] for feat in feats]
            hair_feats = []
            eye_feats = []
            for feat in feats:
                if feat in hair_types:
                    hair_feats.append(feat.split(' ')[0])
                elif feat in eye_types:
                    eye_feats.append(feat.split(' ')[0])
            tags[id] = (hair_feats, eye_feats)
    return tags


def arti_feats(args, colors=None):
    feats = np.zeros([args.batch_size, args.emb_dim*2], dtype=float)
    if colors is None:
        for i in range(args.batch_size):
            colors = ([random.choice(color_list)], [random.choice(color_list)])
            feats[i] = one_hot_feats(colors, args)
    else:
        assert(len(colors) == 2)
        for i in range(args.batch_size):
            feats[i] = one_hot_feats(colors, args)

    return feats