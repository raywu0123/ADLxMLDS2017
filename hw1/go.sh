#!/bin/bash -ex

echo "Usage : ./go.sh [pretrain] [pretest] [train] [test]"

batch_size=128
hidden_size=256
info_epoch=100
init_scale=0.1
keep_prob=1
learning_rate=0.001
decay_steps=20
decay_rate=0.99
log_dir=CNN_logs2
max_epoch=200000
max_grad_norm=1
rnn_layer_num=1
rnn_type=1 # 0: LSTM, 1: GRU
save_model_secs=120
train_file=data/trainframes.npy
n_class=48
window_size=64
dim=69
use_bidirection=''

kernel_size=5
filter_num=128
val_ratio=0.1
vote_num=10

data_dir='./data'

for var in "$@"
do
  if [ "$var" == "pretrain" ]
  then
    python3 preprocess.py \
      --mode            'train' \
      --data_dir        $data_dir   \
      --output_dir      $data_dir
  elif [ "$var" == "pretest" ]
  then
    python3 preprocess.py \
      --mode            'test' \
      --data_dir        $data_dir   \
      --output_dir      $data_dir
  elif [ "$var" == "train" ]
  then
    python3 train_Daikon.py \
      --batch_size      $batch_size  \
      --hidden_size     $hidden_size    \
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
      --save_model_secs  $save_model_secs    \
      --window_size     $window_size \
      --n_class         $n_class \
      --dim             $dim \
      --valid_ratio     $val_ratio \
      --kernel_size     $kernel_size \
      --filter_num      $filter_num  \
      $use_bidirection
  elif [ "$var" == "test" ]
    then
    python3 test.py \
      --batch_size      $batch_size  \
      --hidden_size     $hidden_size    \
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
      --save_model_secs  $save_model_secs    \
      --window_size     $window_size \
      --n_class         $n_class \
      --dim             $dim \
      --kernel_size     $kernel_size \
      --filter_num      $filter_num  \
      --vote_num        $vote_num   \
      $use_bidirection
  fi
done
