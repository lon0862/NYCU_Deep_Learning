import argparse
import itertools
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from dataset import bair_robot_pushing_dataset
from models.cvae import cvae
from utils import finn_eval_seq
import numpy as np
from utils import plot_pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--best_model_dir', default='./logs/best_model', help='base directory to save best model')
    parser.add_argument('--data_root', default='./processed_data', help='root directory for data')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')

    parser.add_argument('--seed', default=2023, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument("--cond_dim", type=int , default=7, help="dimensionality of condition")
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading threads')
    
    args = parser.parse_args()
    return args

def load_ckpt(args, load_ckpt_dir, model, device):
    state = torch.load(load_ckpt_dir, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    print('best model loaded from %s' % load_ckpt_dir)
    return model

def test(test_loader, model, args):
    psnr_list = []
    epoch_loss = 0
    with torch.no_grad():
        for i, (seq, cond) in enumerate(tqdm(test_loader)):
            model.eval()
            seq, cond = seq.to(device), cond.to(device)
            frames_num = args.n_past + args.n_future
            seq = seq.permute(1, 0, 2, 3 ,4)[:frames_num] # [12, batch_size, 3, 64, 64]
            cond = cond.permute(1, 0, 2)[:frames_num] # [12, batch_size, 7]
            pred_seq, mse, kld = model.predict(seq, cond)

            _, _, psnr = finn_eval_seq(seq[args.n_past:], pred_seq[args.n_past:])
            psnr_list.append(psnr)

            if i==0:
                plot_pred(seq, pred_seq, args, device, sample_idx=0)

        ave_psnr = np.mean(np.concatenate(psnr_list))
        print("[Epoch best] test psnr = {:.5f}".format(ave_psnr))

if __name__ == '__main__':
    args = parse_args()
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    
    test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            shuffle=True, drop_last=True, pin_memory=True)
    test_iterator = iter(test_loader)
    model = cvae(args).to(device)
    # --------- load model ------------------------------------
    if args.kl_anneal_cyclical:
        load_ckpt_dir = args.best_model_dir + '/best_model_cyclical.pt'
    else:
        load_ckpt_dir = args.best_model_dir + '/best_model_monotonic.pt'

    model = load_ckpt(args, load_ckpt_dir, model, device)
    # --------- test ------------------------------------
    test(test_loader, model, args)