import math
import operator
import sys
import json
from functools import reduce
import config
import os
import numpy as np

args = config.parse_arguments()

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    epsilon = 1e-10
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/(c+epsilon)))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(s,t):
    score = 0.  
    count = 0
    candidate = [s.strip()]
    references = [[t.strip()]] 
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score


def int2string(int_pred):
    dct = open(os.path.join(args.preprocess_dir, 'vocab.txt'), 'r').read().splitlines()
    word2int = dict([[word, i] for i, word in enumerate(dct)])
    int2word = dict([[i, word] for i, word in enumerate(dct)])
    words = []
    idx = 0
    while idx < len(int_pred) and int_pred[idx] != 2:
        words.append(int2word[int_pred[idx]])
        idx += 1
    return ' '.join(words)


def calc_bleu(output):
    test = json.load(open(os.path.join(args.data_dir,'testing_label.json'),'r'))
    result = {}
    with open(output,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    bleu=[]
    for item in test:
        score_per_video = []
        for caption in item['caption']:
            score_per_video.append(BLEU(result[item['id']],caption))
        bleu.append(sum(score_per_video)/len(score_per_video))
    average = sum(bleu) / len(bleu)
    return average

def get_inference_batch(video_ids):
  batch_vggs = np.zeros([args.batch_size, args.frame_num, args.feat_num])
  batch_captions = np.zeros([args.batch_size, args.max_sent_len], dtype=int)
  batch_lens = np.zeros([args.batch_size], dtype=int)

  for idx, video_name in enumerate(video_ids):
    vgg = np.load(os.path.join(args.data_dir, 'testing_data/feat/' + video_name + '.npy'))
    batch_vggs[idx] = vgg
  return (batch_vggs, batch_captions, batch_lens)


