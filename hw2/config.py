import argparse

# Preprocess Arguments
dft_data_dir = './MLDS_hw2_data'
dft_preprocess_dir = './preprocess'
dft_max_sent_len = 46

dft_vocab_emb_dim = 300
dft_video_emb_dim = 512

dft_output_file = './test_outputs.txt'
dft_model_type = 'seq2seq'
dft_batch_size = 100
dft_frame_num = 80
dft_feat_num = 4096
dft_init_scale = 0.1
dft_log_dir = './logs'
dft_save_model_secs = 120
dft_rnn_type = 1
dft_rnn_layer_num = 1
dft_max_grad_norm = 1
dft_keep_prob = 1.
dft_learning_rate = 0.001
dft_decay_steps = 1000
dft_decay_rate = 0.99
dft_max_epoch = 1200
dft_info_epoch = 1
dft_enc_dim = 512
dft_dec_dim = 512
dft_tao = 200
dft_test_mode = 'testing'

def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', type=str, default=dft_data_dir)
  parser.add_argument('--preprocess_dir', type=str, default=dft_preprocess_dir)
  parser.add_argument('--max_sent_len', type=int, default=dft_max_sent_len)
  parser.add_argument('--vocab_emb_dim', type=int, default=dft_vocab_emb_dim)
  parser.add_argument('--video_emb_dim', type=int, default=dft_video_emb_dim)
  parser.add_argument('--train_wv', action='store_true')

  parser.add_argument('--model_type', type=str, default=dft_model_type)
  parser.add_argument('--batch_size', type=int, default=dft_batch_size)
  parser.add_argument('--frame_num', type=int, default=dft_frame_num)
  parser.add_argument('--feat_num', type=int, default=dft_feat_num)
  parser.add_argument('--init_scale', type=float, default=dft_init_scale)
  parser.add_argument('--log_dir', type=str, default=dft_log_dir)
  parser.add_argument('--save_model_secs', type=int, default=dft_save_model_secs)
  parser.add_argument('--rnn_type', type=int, default=dft_rnn_type,
                         help='rnn cell type: 0->LSTM, 1->GRU '
                              '(default:%(default)s)')
  parser.add_argument('--output_file', type=str, default=dft_output_file)
  parser.add_argument('--rnn_layer_num', type=int, default=dft_rnn_layer_num)
  parser.add_argument('--max_grad_norm', type=int, default=dft_max_grad_norm)
  parser.add_argument('--keep_prob', type=float,default=dft_keep_prob)
  parser.add_argument('--learning_rate', type=float, default=dft_learning_rate)
  parser.add_argument('--decay_steps', type=float, default=dft_decay_steps)
  parser.add_argument('--decay_rate', type=float, default=dft_decay_rate)
  parser.add_argument('--max_epoch', type=int, default=dft_max_epoch)
  parser.add_argument('--info_epoch', type=int, default=dft_info_epoch)
  parser.add_argument('--enc_dim', type=int, default=dft_enc_dim)
  parser.add_argument('--dec_dim', type=int, default=dft_dec_dim)
  parser.add_argument('--tao', type=int, default=dft_tao)
  parser.add_argument('--use_bidirection', action='store_true')
  parser.add_argument('--test_mode', type=str, default=dft_test_mode)
  parser.add_argument('--special', action='store_true')
  parser.add_argument('--aug', action='store_true')

  return parser.parse_args()