def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--max_ep', type=int, default=1000000)
    parser.add_argument('--init_scale', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default='./dqn_logs')
    parser.add_argument('--save_model_secs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    return parser
