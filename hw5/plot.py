import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    args = parser.parse_args()
    return args

# read txt file
def read_txt(train_log_dir, valid_log_dir, title):
    epoch_list = []
    train_loss_list = []
    train_mse_list = []
    train_kld_list = []
    tfr_list = []
    beta_list = []
    psnr_list = []

    with open(train_log_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            numbers = re.findall(r'\d+\.\d+|\d+', line)
            epoch_list.append(int(numbers[0]))
            train_loss_list.append(float(numbers[1]))
            train_mse_list.append(float(numbers[2]))
            train_kld_list.append(float(numbers[3]))
            tfr_list.append(float(numbers[4]))
            beta_list.append(float(numbers[5]))
    
    with open(valid_log_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            numbers = re.findall(r'\d+\.\d+|\d+', line)
            psnr_list.append(float(numbers[2]))

    fig, ax1 = plt.subplots()
    plt.title(title)
    plt.xlabel('loss/epochs') 
    ax2 = ax1.twinx()
    ax1.set_ylabel('loss/psnr')
    ax1.scatter(epoch_list, psnr_list, c='g', s=2, label='psnr')
    ax1.plot(epoch_list, train_kld_list, 'b', label='kld')
    ax1.plot(epoch_list, train_loss_list, 'brown', label='loss')
    ax1.plot(epoch_list, train_mse_list,'r', label='mse')
    ax1.legend()
    ax1.set_ylim(0, 30)

    ax2.set_ylabel('ratio')
    ax2.plot(epoch_list, tfr_list, 'm--', label='Teacher ratio')
    ax2.plot(epoch_list, beta_list, '--', color='orange', label='KL beta')
    fig.tight_layout()
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    plt.show()

    
if __name__ == '__main__':
    args = parse_args()
    if args.kl_anneal_cyclical:
        train_log_dir = args.log_dir + '/train_record_cyclical.txt'
        valid_log_dir = args.log_dir + '/valid_record_cyclical.txt'
        title = "Cyclical KL Annealing"
    else:
        train_log_dir = args.log_dir + '/train_record_monotonic.txt'
        valid_log_dir = args.log_dir + '/valid_record_monotonic.txt'
        title = "Monotonic KL Annealing"

    read_txt(train_log_dir, valid_log_dir, title)