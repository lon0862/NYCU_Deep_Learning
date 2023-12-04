import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.cvae import cvae
from utils import finn_eval_seq, record_train, record_val, save_ckpt, load_ckpt

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='./logs/models', help='base directory to save models')
    parser.add_argument('--best_model_dir', default='./logs/best_model', help='base directory to save best model')
    parser.add_argument('--data_root', default='./processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=150, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
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
    parser.add_argument('--beta', type=float, default=0, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

    args = parser.parse_args()
    return args

class kl_annealing():
    def __init__(self, args, beta):
        super().__init__()
        self.args = args
        self.beta = beta
        self.kl_anneal_cyclical = self.args.kl_anneal_cyclical
        self.kl_anneal_cycle = self.args.kl_anneal_cycle
        if self.kl_anneal_cyclical:
            self.period = self.args.niter // self.kl_anneal_cycle
            self.step = 1 / (self.period // 2)
        else:
            self.period = self.args.niter // 8
            self.step = self.args.kl_anneal_ratio / self.period

    def update(self, epoch):
        if self.kl_anneal_cyclical:
            if (epoch % self.period) <= (self.period // 2):
                if epoch % self.period == 0:
                    self.beta = self.args.beta
                else:
                    self.beta += self.step
                    self.beta = min(self.beta, 1.0)
        else:
            if epoch < (self.period * 2) :
                self.beta += self.step
                self.beta = min(self.beta, 1.0)
    
    def get_beta(self):
        return self.beta

def train(train_loader, train_iterator, model, optimizer, args, tfr, beta):
    epoch_loss = 0
    epoch_mse = 0
    epoch_kld = 0
    for _ in tqdm(range(args.epoch_size)):
        try:
            ## Train on next batch
            seq, cond = next(train_iterator)
        except StopIteration:
            ## If all batches have been trained, return to the first batch
            train_iterator = iter(train_loader)
            seq, cond = next(train_iterator)

        model.train()
        seq, cond = seq.to(device), cond.to(device)
        seq = seq.permute(1, 0, 2, 3 ,4) # [n_frames, batch_size, 3, 64, 64]
        cond = cond.permute(1, 0, 2) # [n_frames, batch_size, 7]

        # calculate loss
        mse, kld = model(seq, cond, tfr)
        loss = mse + kld * beta
        epoch_loss += (loss.item() / (args.n_past + args.n_future))
        epoch_mse += (mse.item() / (args.n_past + args.n_future))
        epoch_kld += (kld.item() / (args.n_future + args.n_past))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= args.epoch_size
    epoch_mse /= args.epoch_size
    epoch_kld /= args.epoch_size       

    return epoch_loss, epoch_mse, epoch_kld

def val(valid_data, valid_loader, valid_iterator, model, args, beta):
    psnr_list = []
    epoch_loss = 0
    valid_epoch_size = len(valid_data) // args.batch_size
    with torch.no_grad():
        for _ in range(valid_epoch_size):
            try:
                ## Train on next batch
                seq, cond = next(valid_iterator)
            except StopIteration:
                ## If all batches have been trained, return to the first batch
                valid_iterator = iter(valid_loader)
                seq, cond = next(valid_iterator)

            model.eval()
            seq, cond = seq.to(device), cond.to(device)
            seq = seq.permute(1, 0, 2, 3 ,4) # [12, batch_size, 3, 64, 64]
            cond = cond.permute(1, 0, 2) # [12, batch_size, 7]

            pred_seq, mse, kld = model.predict(seq, cond)
            loss = mse + kld * beta
            epoch_loss += (loss.item() / (args.n_past + args.n_future))

            _, _, psnr = finn_eval_seq(seq[args.n_past:], pred_seq[args.n_past:])
            psnr_list.append(psnr)
        
        epoch_loss /= valid_epoch_size
        ave_psnr = np.mean(np.concatenate(psnr_list))
    
    return epoch_loss, ave_psnr

if __name__ == '__main__':
    args = parse_args()
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir + '/cyclical', exist_ok=True)
    os.makedirs(args.model_dir + '/monotonic', exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    valid_data = bair_robot_pushing_dataset(args, 'validate')

    train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            shuffle=True, drop_last=True, pin_memory=True)
    train_iterator = iter(train_loader)

    valid_loader = DataLoader(valid_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            shuffle=True, drop_last=True, pin_memory=True)
    valid_iterator = iter(valid_loader)
    # ---------------- model and optimizers ----------------
    model = cvae(args).to(device)
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    optimizer = args.optimizer(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # ----------------- init reference -----------------
    tfr = args.tfr
    tfr_decay_step = (args.tfr - args.tfr_lower_bound) / (args.niter - args.tfr_start_decay_epoch)
    start_epoch = 0
    best_val_psnr = 0
    beta = args.beta
    kl_anneal = kl_annealing(args, beta)
    # --------- load model ------------------------------------
    if args.kl_anneal_cyclical:
        load_ckpt_dir = args.model_dir + '/cyclical'
    else:
        load_ckpt_dir = args.model_dir + '/monotonic'

    all_file_name = os.listdir(load_ckpt_dir)
    num = 0
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            tmp = all_file_name[i].split('.')
            tmp = tmp[0].split('_')
            if(num<int(tmp[1])):
                num=int(tmp[1])
        load_ckpt_dir += '/ckpt_' + str(num) + '.pt'
        model, optimizer, start_epoch, beta, tfr, best_val_psnr = load_ckpt(args, load_ckpt_dir, model, optimizer)
    # --------- training loop ------------------------------------
    best_val_psnr = 0
    for epoch in range(start_epoch, start_epoch + args.niter):
        beta = kl_anneal.get_beta()

        train_loss, train_mse, train_kld = \
            train(train_loader, train_iterator, model, optimizer, args, tfr, beta)
        record_train(args, epoch,  train_loss, train_mse, train_kld, beta, tfr)

        val_loss, ave_psnr = val(valid_data, valid_loader, valid_iterator, model, args, beta)
        best_val_psnr = record_val(val_loss, ave_psnr, best_val_psnr, model, optimizer, args, epoch, beta, tfr)
        
        ## Update KL annealing weight
        kl_anneal.update(epoch)

        ### Update teacher forcing ratio ###
        if epoch >= args.tfr_start_decay_epoch:
            tfr -= tfr_decay_step
            tfr = max(tfr, args.tfr_lower_bound)
        
        # save the model
        if (epoch + 1) % 10 == 0:
            if args.kl_anneal_cyclical:
                ckpt_dir = args.model_dir + '/cyclical/ckpt_' + str(epoch+1) + '.pt'
            else:
                ckpt_dir = args.model_dir + '/monotonic/ckpt_' + str(epoch+1) + '.pt'

            save_ckpt(ckpt_dir, model, optimizer, args, epoch, beta, tfr, best_val_psnr)
            print('model saved to %s' % ckpt_dir)