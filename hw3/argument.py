def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    # parser.add_argument('--max_episode', type=int, default=1000000)
    # parser.add_argument('--max_step', type=int, default=1e7)
    # parser.add_argument('--start_step', type=int, default=10000)
    # parser.add_argument('--info_step', type=int, default=1000)
    # parser.add_argument('--max_epoch', type=int, default=5)
    # parser.add_argument('--train_freq', type=int, default=4)
    # parser.add_argument('--tar_update_freq', type=int, default=1000)
    # parser.add_argument('--init_scale', type=float, default=0.01)
    # parser.add_argument('--log_dir', type=str, default='./dqn_logs')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    # parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.99)
    # parser.add_argument('--memSize', type=int, default=10000)
    # parser.add_argument('--GAMMA', type=float, default=0.99)
    # parser.add_argument('--explore_rate', type=float, default=0.1)
    # parser.add_argument('--screen_height', type=int, default=84)
    # parser.add_argument('--memLog', type=str, default='./dqn_logs/memLog')

    parser.add_argument('--max_episode', type=int, default=10000)
    parser.add_argument('--log_dir', type=str, default='./pg_logs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--GAMMA', type=float, default=0.99)

    return parser
