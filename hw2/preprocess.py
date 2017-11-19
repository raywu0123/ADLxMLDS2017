#!/usr/bin/python3
import os, argparse, json, re
import numpy as np
from gensim.models import word2vec
from collections import defaultdict
from tqdm import tqdm
import tensorflow as tf
import config
import pickle
from utils import int2string
# Generate int_caption.pkl s

args = config.parse_arguments()

def process_sent(sent):
  s = sent.lower()
  for deli in ['\'','.','?','!',',',';',':','\"', '(', ')']:
    s = re.sub('['+deli+']', ' '+deli, s)
  return s.split()

if not os.path.exists(args.preprocess_dir):
  os.makedirs(args.preprocess_dir)

def load_dict():
  dict={}
  with open(os.path.join(args.preprocess_dir, 'vocab.txt'), 'r', encoding='utf8') as file:
    for id, line in enumerate(file):
      dict[line.strip('\n')] = id
  return dict

def train_wv():
  sent_lens = []
  all_words = []
  all_sents = []

  with open(os.path.join(args.data_dir, 'training_label.json'), 'r') as label_file:
    labels = json.load(label_file)
    for label in labels:
      for sent in label['caption']:
        pro_sent = process_sent(sent)
        sent_lens.append(len(pro_sent))
        all_words.extend(pro_sent)
        all_sents.append(['BOS'] + pro_sent + ['EOS'])

  print('training word2vec...')
  model = word2vec.Word2Vec(all_sents, size=args.vocab_emb_dim, window=5, min_count=0, workers=4)
  wv = []

  print('building dictionary...')
  vocabs = list(set(all_words))
  PAD, BOS, EOS, UNK = 0, 1, 2, 3
  vocabs = ['PAD', 'BOS', 'EOS', 'UNK'] + [vocab for vocab in vocabs]
  print(len(vocabs))
  dct = defaultdict(lambda: UNK)
  with open(os.path.join(args.preprocess_dir, 'vocab.txt'), 'w', encoding='utf8') as f:
    for i, vocab in enumerate(vocabs):
      f.write(vocab + '\n')
      dct[vocab] = i
      if vocab == 'PAD' or vocab == 'UNK':
        wv.append(np.zeros((args.vocab_emb_dim)))
      else:
        wv.append(model.wv[vocab])
  wv = np.array(wv)
  print('size of wv = {}'.format(wv.shape))
  np.save(os.path.join(args.preprocess_dir, 'wv.npy'), wv)

if args.train_wv:
  train_wv()

dict = load_dict()
modes = ['training', 'testing']

for mode in modes:
  with open(os.path.join(args.data_dir, mode+'_label.json')) as json_file:
    labels = json.load(json_file)
    filename_captions_map = {}
    unk_count = 0
    word_count = 0
    for label in tqdm(labels):
      int_captions = []
      caption_lens = []
      for caption in label['caption']:
        pro_sent = ['BOS'] + process_sent(caption) + ['EOS']
        int_caption = np.zeros(shape=[args.max_sent_len], dtype=int)
        for idx, word in enumerate(pro_sent):
          word_count += 1
          if word not in dict:
              int_caption[idx] = dict['UNK']
              unk_count += 1
          else:
            int_caption[idx] = dict[word]
        int_captions.append((int_caption, len(pro_sent)))

      filename_captions_map[label['id']] = int_captions

    print('unkown/num_words= ', unk_count/float(word_count))
    with open(os.path.join(args.preprocess_dir, mode+'_int_captions.pkl'), 'wb+') as dump_file:
      pickle.dump(filename_captions_map, dump_file)