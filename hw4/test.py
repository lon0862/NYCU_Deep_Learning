import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ResNet_18, ResNet_50
import torchvision.models as models
from dataloader import RetinopathyLoader
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    labels = [0, 1, 2, 3, 4]
    cm= confusion_matrix(y_true, y_pred,labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        include_values=True,            
        cmap=plt.cm.Blues,                                        
        xticks_rotation="horizontal"              
    )
    disp.ax_.set_title(title)
    plt.savefig(save_path)
    plt.show()

def test(dataloader, model, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_accuracy = 0
    labels = []
    pred = []
    total_num = 0
    with torch.no_grad():
        for data_list in tqdm(dataloader):
            model.eval()
            data, label = data_list
            data = data.to(device)
            label = label.to(device)

            out = model(data)
            pred_label = torch.argmax(out, dim=1)

            labels.extend(label.detach().cpu().numpy().tolist())
            pred.extend(pred_label.detach().cpu().numpy().tolist())
  
            total_accuracy += (pred_label==label).sum().item()
            total_num += pred_label.shape[0]

    total_accuracy /= total_num

    return total_accuracy, labels, pred



def load_ckpt(ckpt_dir, model, device):
    state = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    print('model loaded from %s' % ckpt_dir)
    return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--batch_size', type=int, default=16) # resnet50 use 8
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='ResNet_18')
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--mode', type=str, default='train')
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

    if args.model_name=='ResNet_18':
        if args.pretrained==True:
            print("model is pretrained Resnet18")
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 5)
            dir_path += '/ckpt/pretrained/resnet18'
            save_path = 'cm_img/pretrained/resnet18'
            title = 'pretrained_resnet18'
        else:
            print("model is no_pretrained Resnet18")
            model = ResNet_18()
            dir_path += '/ckpt/no_pretrained/resnet18'
            save_path = 'cm_img/no_pretrained/resnet18'
            title = 'no_pretrained_resnet18'
    else:
        if args.pretrained==True:
            print("model is pretrained Resnet50")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 5)
            dir_path += '/ckpt/pretrained/resnet50'
            save_path = 'cm_img/pretrained/resnet50'
            title = 'pretrained_resnet50'
        else:
            print("model is no_pretrained Resnet50")
            model = ResNet_50()
            dir_path += '/ckpt/no_pretrained/resnet50'
            save_path = 'cm_img/no_pretrained/resnet50'
            title = 'no_pretrained_resnet50'
        
    model = model.to(device)
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
        model = load_ckpt(load_ckpt_dir, model, device)
    # ================================== LOAD CKPT =========================================
    save_path = save_path + '_' + args.mode + '.png'
    title = 'CM_' + title + '_' + args.mode
    if args.mode == 'train':
        train_accuracy, labels, pred = test(train_dataloader, model, device)
        print("train_accuracy: %.6f" % train_accuracy)
        plot_confusion_matrix(labels, pred, title, save_path)
    else:
        test_accuracy, labels, pred = test(test_dataloader, model, device)
        print("test_accuracy: %.6f" % test_accuracy)
        plot_confusion_matrix(labels, pred, title, save_path)