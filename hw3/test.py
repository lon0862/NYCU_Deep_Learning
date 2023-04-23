import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.model import EEGNet, DeepConvNet
from argparse import ArgumentParser
from dataloader import read_bci_data
from torch.utils.data import TensorDataset, DataLoader

def test(dataloader, model, device):
    total_loss = 0
    total_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    for i, data_list in enumerate(dataloader):
        model.eval()
        data, label = data_list
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.int64)

        out = model(data)
        pred_label = torch.argmax(out, dim=1)

        loss = criterion(out, label)
        total_loss += loss.item()
        accuracy = (pred_label==label).sum() / pred_label.shape[0]
        total_accuracy += accuracy.item()

    total_loss /= len(dataloader) 
    total_accuracy /= len(dataloader)  

    return total_loss, total_accuracy

def load_ckpt(ckpt_dir, model):
    state = torch.load(ckpt_dir)
    model.load_state_dict(state['model_state_dict'])
    
    print('model loaded from %s' % ckpt_dir)
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    # ==================== INIT DATASET ================================
    '''
    train_data: (1080, 1, 2, 750)
    train_label: (1080, )
    test_data: (1080, 1, 2, 750)
    test_label: (1080, )
    '''
    train_data, train_label, test_data, test_label = read_bci_data()
    
    # 轉成 torch tensor
    test_data_tensor = torch.from_numpy(test_data)
    test_label_tensor = torch.from_numpy(test_label)

    # 建立 dataset
    test_dataset = TensorDataset(test_data_tensor, test_label_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # ==================== INIT DATASET =================================
    dir_path = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    act_model_list = ["ReLU", "Leaky_ReLU", "ELU"]
    for act_model_name in act_model_list:
        if(act_model_name=="ReLU"):
            act_model = nn.ReLU()
        elif(act_model_name=="Leaky_ReLU"):
            act_model = nn.LeakyReLU()
        elif(act_model_name=="ELU"):
            act_model = nn.ELU()

        if args.model=='EEGNet':
            model = EEGNet(act_model).to(device)
        elif args.model=='DeepConvNet':
            model = DeepConvNet(act_model).to(device)

        ckpt_dir = dir_path + "/" + args.model + "_ckpt/" +act_model_name + ".pt"
        model = load_ckpt(ckpt_dir, model)
        
        test_loss, test_acc = test(test_dataloader, model, device)
        test_acc *= 100
        print("best test accuracy: %.6f" % test_acc)
