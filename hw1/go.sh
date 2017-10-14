#!/bin/bash -ex

echo "Usage : ./go.sh [pretrain] [pretest] [train] [test]"

batch_size=512
hidden_size=256
info_epoch=10
init_scale=0.1
keep_prob=1
learning_rate=0.001
decay_steps=20
decay_rate=0.99
log_dir=logs
max_epoch=200000
max_grad_norm=1
rnn_layer_num=1
rnn_type=1 # 0: LSTM, 1: GRU
save_model_secs=120
train_file=data/trainframes.npy
n_class=48
window_size=128
dim=69
use_bidirection=''

kernel_size=3
filter_num=128

val_ratio=0.1
train_ark_path='./data/fbank/train.ark'
test_ark_path='./data/fbank/test.ark'
lab_path='./data/train.lab'
preprocess_output_path='./data/'

for var in "$@"
do
  if [ "$var" == "pretrain" ]
  then
    python3 preprocess.py \
      --mode            'train' \
      --ark             $train_ark_path   \
      --lab             $lab_path   \
      --output_dir      $preprocess_output_path
  elif [ "$var" == "pretest" ]
  then
    python3 preprocess.py \
      --mode            'test' \
      --ark             $test_ark_path   \
      --output_dir      $preprocess_output_path
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
      --train_file      $train_file   \
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
      --train_file      $train_file   \
      --window_size     $window_size \
      --n_class         $n_class \
      --dim             $dim \
      --kernel_size     $kernel_size \
      --filter_num      $filter_num  \
      $use_bidirection
  fi
done