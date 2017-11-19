#!/bin/bash -ex

data_dir=$1
test_output_file=$2
peer_output_file=$3
log_dir='./hw2_FINAL_logs/logs_FINAL'
preprocess_dir='./preprocess'

vocab_emb_dim=300
video_emb_dim=512
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
rnn_type=0 # 0: LSTM, 1: GRU
save_model_secs=120
model_type='seq2seq'

echo 'Cloning logs'
wget https://gitlab.com/raywu0123/hw2_FINAL_logs/repository/master/archive.tar.gz

mv hw2_FINAL_logs-master* hw2_FINAL_logs

#echo 'Preprocessing'
#    python3 preprocess.py \
#        --data_dir      $data_dir   \
#        --preprocess_dir    $preprocess_dir \
#        --vocab_emb_dim       $vocab_emb_dim    \
#        --max_sent_len  $max_sent_len

echo 'Running Peer-Review mode'
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
      --output_file     $peer_output_file    \
      --test_mode       'peer_review'

echo 'Running Testing mode'
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
      --output_file     $test_output_file    \
      --test_mode       'testing'