#!/bin/bash -ex

echo "Usage : ./go.sh [preprocess] [train] [test]"

data_dir='./MLDS_hw2_data'
log_dir='./logs_2'
preprocess_dir='./preprocess'

vocab_emb_dim=300
video_emb_dim=512
max_sent_len=46
batch_size=100
info_epoch=1
init_scale=0.1
keep_prob=1
learning_rate=0.001
decay_steps=50
decay_rate=0.99

max_epoch=1000
max_grad_norm=1
rnn_layer_num=1
rnn_type=1 # 0: LSTM, 1: GRU
save_model_secs=120
model_type='S2VT'


for var in "$@"
do
  if [ "$var" == "preprocess" ]
  then
    python3 preprocess.py \
        --data_dir      $data_dir   \
        --preprocess_dir    $preprocess_dir \
        --vocab_emb_dim       $vocab_emb_dim    \
        --max_sent_len  $max_sent_len
  elif [ "$var" == "train" ]
  then
    python3 train.py \
      --data_dir        $data_dir   \
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
      --'aug'
  elif [ "$var" == "test" ]
  then
    python3 test.py \
      --data_dir      $data_dir   \
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
      --model_type      $model_type
  fi
done
