#!/bin/bash -ex

echo "Usage : ./go.sh [pretrain] [pretest] [train] [test]"

#sorted alphabetically
batch_size=512
hidden_size=64
info_epoch=10
init_scale=0.01
keep_prob=1
learning_rate=0.001
decay_steps=20
decay_rate=1.0
log_dir=logs
max_epoch=200000
max_grad_norm=100
rnn_layer_num=1
rnn_type=1 # 0: LSTM, 1: GRU
save_model_secs=120
train_file=data/trainframes.npy
n_class=48
window_size=128
dim=69
use_bidirection=''



for var in "$@"
do
  if [ "$var" == "train" ]
  then
    ./train_Daikon.py \
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
      --train_file      $train_file   \
      --window_size     $window_size \
      --n_class         $n_class \
      --dim             $dim \
      $use_bidirection
  elif [ "$var" == "test" ]
    then
    ./test.py \
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
      --train_file      $train_file   \
      --window_size     $window_size \
      --n_class         $n_class \
      --dim             $dim \
      $use_bidirection
  fi
done