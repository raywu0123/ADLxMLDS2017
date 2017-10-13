import argparse

dft_rnn_type = 1 # 0: LSTM, 1: GRU
dft_hidden_size = 64
dft_batch_size = 512
dft_rnn_layer_num = 1
dft_max_grad_norm = 1
dft_keep_prob = 1
dft_init_scale = 0.001
dft_max_epoch = 10
dft_info_epoch = 10
dft_save_model_secs = 120
dft_learning_rate = 0.0001
dft_decay_steps=10000
dft_decay_rate= 0.99
dft_train_file = 'data/trainframes.npy'
dft_log_dir = 'logs'
dft_n_class = 48
dft_window_size = 128
dft_dim = 69
dft_valid_ratio = 0.1

dft_get_weights = True
dft_weights_file = 'RNN_weights.pickle'
dft_cnn_layer_num = 1
dft_test_file = 'data/testframes.npy'
dft_pred_file = 'submission/pred.csv'
dft_filter_num = 12
dft_kernel_size = 2
dft_pool_size = 2
dft_test_num = 500 # number of test data

def parse_arguments():

  argparser = argparse.ArgumentParser(description='Sequential Matching Network')

  argparser.add_argument('-rct', '--rnn_type', type=int, default=dft_rnn_type,
    										 help='rnn cell type: 0->LSTM, 1->GRU '
                         '(default:%(default)s)')
  argparser.add_argument('-hu', '--hidden_size', type=int,
                         default=dft_hidden_size, help='hidden units of rnn '
                         'cell (default:%(default)s)')
  argparser.add_argument('-bs', '--batch_size', type=int,
                         default=dft_batch_size,
                         help='batch size (default:%(default)s)')
  argparser.add_argument('-cln', '--cnn_layer_num', type=int,
                         default=dft_cnn_layer_num,
                         help='number of cnn layers (default:%(default)s)')
  argparser.add_argument('-rln', '--rnn_layer_num', type=int,
                         default=dft_rnn_layer_num,
                         help='number of rnn layers (default:%(default)s)')
  argparser.add_argument('-fn', '--filter_num', type=int,
                         default=dft_filter_num,
                         help='number of filters (default:%(default)s)')
  argparser.add_argument('-ks', '--kernel_size', type=int,
                         default=dft_kernel_size,
                         help='size of kernel for 2d cnn (default:%(default)s)')
  argparser.add_argument('-ps', '--pool_size', type=int,
                         default=dft_pool_size,
                         help='size of max pooling (default:%(default)s)')
  argparser.add_argument('-mgn', '--max_grad_norm', type=int,
                         default=dft_max_grad_norm,
                         help='maximum gradient norm (default:%(default)s)')
  argparser.add_argument('-kp', '--keep_prob', type=float,
                         default=dft_keep_prob, help='keep probability '
                         'of dropout layer (default:%(default)s)')
  argparser.add_argument('-lr', '--learning_rate', type=float,
                         default=dft_learning_rate, help='learning rate '
                         '(default:%(default)s)')
  argparser.add_argument('-ds', '--decay_steps', type=float,
                        default=dft_decay_steps, help='decay steps '
                        '(default:%(default)s)')
  argparser.add_argument('-dr', '--decay_rate', type=float, 
                        default=dft_decay_rate, help='decay rate '
                        '(default:%(default)s)')
  argparser.add_argument('-is', '--init_scale', type=float,
                         default=dft_init_scale, help='initialization scale for'
                         ' tensorflow initializer (default:%(default)s)')
  argparser.add_argument('-me', '--max_epoch', type=int, default=dft_max_epoch,
                         help='maximum training epoch '
                         '(default:%(default)s)')
  argparser.add_argument('-ie', '--info_epoch', type=int,
                         default=dft_info_epoch, help='show training '
                         'information for each (default:%(default)s) epochs')
  argparser.add_argument('-tn', '--test_num', type=int,
                         default=dft_test_num, help='number of test '
                         'data (default:%(default)s)')
  argparser.add_argument('-sms', '--save_model_secs', type=int,
                         default=dft_save_model_secs, help='save model for '
                         'every SAVE_MODEL_SECS seconds (default:%(default)s)')
  argparser.add_argument('-trf', '--train_file', type=str,
                         default=dft_train_file, help='input training filename '
                         '(default:%(default)s)')
  argparser.add_argument('-tef', '--test_file', type=str,
                         default=dft_test_file, help='test filename '
                         '(default:%(default)s)')
  argparser.add_argument('-pf', '--pred_file', type=str,
                         default=dft_pred_file, help='filename for submission '
                         '(default:%(default)s)')
  argparser.add_argument('-ld', '--log_dir', type=str,
                         default=dft_log_dir, help='log directory '
                         '(default:%(default)s)')
  argparser.add_argument('-ub', '--use_bidirection', action='store_true',
                          help='use bidirectional rnn (default:False)')
  argparser.add_argument('--valid_ratio', type=float,
                          default=dft_valid_ratio, help='Ratio of validation data (default: %(default)s)')
  argparser.add_argument('--get_weights', type=bool,
                          default=dft_get_weights, help = 'Program will output weights of all options if True.')
  argparser.add_argument('--weights_file', type=str,
                         default=dft_weights_file, help='default name of weights file.')
  argparser.add_argument('--n_class', type=int, default=dft_n_class)
  argparser.add_argument('--window_size', type=int, default=dft_window_size)
  argparser.add_argument('--dim', type=int, default=dft_dim)

  return argparser.parse_args()