import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--test_txt', type=str, default='./test_txt.txt')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_img_dir', type=str, default='./imgs')
    parser.add_argument('--save_model_secs', type=int, default=120)

    parser.add_argument('--emb_dim', type=int, default=13)
    parser.add_argument('--noise_dim', type=int, default=100)
    parser.add_argument('--noise_amp', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=1000000)
    parser.add_argument('--info_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--init_scale', type=float, default=1e-1)
    parser.add_argument('--dg_ratio', type=int, default=2)

    return parser.parse_args()