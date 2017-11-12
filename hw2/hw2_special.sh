#!/bin/bash -ex

data_dir=$1
output_file=$2
log_dir='./ADLxMLDS_Hw2/seq2seq_logs'

vocab_emb_dim=300
video_emb_dim=1024
max_sent_len=46
batch_size=100
info_epoch=1
init_scale=0.1
keep_prob=1
learning_rate=0.001
decay_steps=20
decay_rate=0.99

max_epoch=200000
max_grad_norm=1
rnn_layer_num=1
rnn_type=1 # 0: LSTM, 1: GRU
save_model_secs=120
model_type='seq2seq'


echo
git clone https://gitlab.com/raywu0123/ADLxMLDS_Hw2.git

echo 'Running inference for Special Mission'

python3 test.py \
      --batch_size      $batch_size  \
      --info_epoch      $info_epoch    \
      --init_scale      $init_scale \
      --keep_prob       $keep_prob  \
      --learning_rate   $learning_rate \
      --decay_steps     $decay_steps \
      --decay_rate      $decay_rate \
      --log_dir         $log_dir   \
      --max_epoch       $max_epoch \
      --max_grad_norm   $max_grad_norm  \
      --rnn_layer_num   $rnn_layer_num  \
      --rnn_type        $rnn_type  \
      --save_model_secs $save_model_secs    \
      --vocab_emb_dim   $vocab_emb_dim  \
      --video_emb_dim   $video_emb_dim  \
      --model_type      $model_type     \
      --data_dir        $data_dir   \
      --output_file     $output_file    \
      '--special'