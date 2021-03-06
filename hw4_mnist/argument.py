import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_img_dir', type=str, default='./imgs')
    parser.add_argument('--save_model_secs', type=int, default=120)

    parser.add_argument('--noise_dim', type=int, default=100)
    parser.add_argument('--noise_amp', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=1000000)
    parser.add_argument('--info_epoch', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--init_scale', type=float, default=1e-2)
    parser.add_argument('--gp_scale', type=float, default=10.0)

    return parser.parse_args()