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

def train(dataloader, model, device, optimizer):
    total_loss = 0
    total_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    for i, data_list in enumerate(dataloader):
        model.train()
        data, label = data_list
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.int64)

        out = model(data)
        pred_label = torch.argmax(out, dim=1)

        loss = criterion(out, label)
        total_loss += loss.item()
        accuracy = (pred_label==label).sum() / pred_label.shape[0]
        total_accuracy += accuracy.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(dataloader) 
    total_accuracy /= len(dataloader)  

    return total_loss, total_accuracy

def test(dataloader, model, device, optimizer):
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

def save_ckpt(ckpt_dir, model, test_acc):
    state = {
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc
    }

    torch.save(state, ckpt_dir)
    print('model saved to %s' % ckpt_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
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
    train_data_tensor = torch.from_numpy(train_data)
    train_label_tensor = torch.from_numpy(train_label)
    test_data_tensor = torch.from_numpy(test_data)
    test_label_tensor = torch.from_numpy(test_label)

    # 建立 dataset
    train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_label_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # ==================== INIT DATASET =================================
    dir_path = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_rate = 1e-3
    epochs = args.epochs
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

        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
        train_acc_list = []
        test_acc_list = []
        best_acc = 0
        
        for i in tqdm(range(epochs)):          
            train_loss, train_acc = train(train_dataloader, model, device, optimizer)
            test_loss, test_acc = test(test_dataloader, model, device, optimizer)

            train_acc_list.append(train_acc*100)
            test_acc_list.append(test_acc*100)
            if(best_acc<(test_acc*100)):
                best_acc = test_acc*100
                best_model = copy.deepcopy(model)
                
        print("best test accuracy: %.6f" % best_acc)
        save_ckpt_dir = dir_path + "/" + args.model + "_ckpt/" +act_model_name + ".pt"
        save_ckpt(save_ckpt_dir, best_model, best_acc)

        if(act_model_name=="ReLU"):
            plt.plot(range(1, epochs+1), train_acc_list, color='yellow', label='relu_train')
            plt.plot(range(1, epochs+1), test_acc_list, color='blue', label='relu_test')
        elif(act_model_name=="Leaky_ReLU"):
            plt.plot(range(1, epochs+1), train_acc_list, color='green', label='leaky_relu_train')
            plt.plot(range(1, epochs+1), test_acc_list, color='red', label='leaky_relu_test')
        elif(act_model_name=="ELU"):
            plt.plot(range(1, epochs+1), train_acc_list, color='purple', label='elu_train')
            plt.plot(range(1, epochs+1), test_acc_list, color='orange', label='elu_test')

    plt.xlabel('epoch') # 設定 x 軸標題
    plt.ylabel('Accuracy(%)') # 設定 y 軸標題
    plt.legend()
    plt.show()
