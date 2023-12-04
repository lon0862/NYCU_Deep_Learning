import torch
from argparse import ArgumentParser
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import os
import torchvision.models as models
from model import ResNet_18, ResNet_50
import torch.nn as nn
from tqdm import tqdm
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def train(dataloader, model, optimizer, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_accuracy = 0
    for data_list in tqdm(dataloader):
        model.train()
        data, label = data_list
        data = data.to(device)
        label = label.to(device)
        '''
        data: [batch_size, 3, 512, 512]
        label: [batch_size]
        '''
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

def test(dataloader, model, optimizer, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data_list in tqdm(dataloader):
            model.eval()
            data, label = data_list
            data = data.to(device)
            label = label.to(device)

            out = model(data)
            pred_label = torch.argmax(out, dim=1)

            loss = criterion(out, label)
            total_loss += loss.item()
            accuracy = (pred_label==label).sum() / pred_label.shape[0]
            total_accuracy += accuracy.item()

    total_loss /= len(dataloader)
    total_accuracy /= len(dataloader)

    return total_loss, total_accuracy

def save_ckpt(ckpt_dir, model, optimizer, train_loss, train_accuracy, test_loss, test_accuracy):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy, 
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }

    torch.save(state, ckpt_dir)
    print('model saved to %s' % ckpt_dir)

def load_ckpt(ckpt_dir, model, optimizer, device):
    state = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    print('model loaded from %s' % ckpt_dir)
    return model, optimizer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--batch_size', type=int, default=16) # resnet50 use 8
    parser.add_argument('--epochs', type=int, default=10) # resnet50 use 5
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='ResNet_18')
    parser.add_argument('--pretrained', type=str2bool, default=True)
    args = parser.parse_args()

    # ================================== INIT DATASET =====================================
    train_dataset = RetinopathyLoader(args.root, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_dataset = RetinopathyLoader(args.root, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=args.pin_memory)
    # ================================== INIT DATASET =====================================
    # ================================== INIT MODEL =======================================
    dir_path = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    lr_rate = 1e-4 # origin 1e-3
    epochs = args.epochs
    if args.model_name=='ResNet_18':
        if args.pretrained==True:
            print("model is pretrained Resnet18")
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 5)
            dir_path += '/ckpt/pretrained/resnet18'
        else:
            print("model is no_pretrained Resnet18")
            model = ResNet_18()
            dir_path += '/ckpt/no_pretrained/resnet18'
    else:
        if args.pretrained==True:
            print("model is pretrained Resnet50")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 5)
            dir_path += '/ckpt/pretrained/resnet50'
        else:
            print("model is no_pretrained Resnet50")
            model = ResNet_50()
            dir_path += '/ckpt/no_pretrained/resnet50'

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=5e-4)
    # ================================== INIT MODEL =======================================
    # ================================== LOAD CKPT ========================================
    all_file_name = os.listdir(dir_path)
    num = 0
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            tmp = all_file_name[i].split('.')
            tmp = tmp[0].split('_')
            if(num<int(tmp[1])):
                num=int(tmp[1])
        load_ckpt_dir = dir_path + '/ckpt_' + str(num) + '.pt'
        model, optimizer = load_ckpt(load_ckpt_dir, model, optimizer, device)
    # ================================== LOAD CKPT =========================================
    
    for i in range(epochs):
        train_loss, train_accuracy = train(train_dataloader, model, optimizer, device)
        test_loss, test_accuracy = test(test_dataloader, model, optimizer, device)
        print('[epoch %d] train loss: %.6f' %(i + 1, train_loss), "min_accuracy: %.6f" % train_accuracy)
        print('[epoch %d] test loss: %.6f' %(i + 1, test_loss), "min_accuracy: %.6f" % test_accuracy)
        save_ckpt_dir = dir_path + '/ckpt_' + str(num+i+1) + '.pt'
        save_ckpt(save_ckpt_dir, model, optimizer, train_loss, train_accuracy, test_loss, test_accuracy)
    


