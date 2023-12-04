import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    dir_path = "ckpt/pretrained/resnet50/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    all_file_name = os.listdir(dir_path)
    epochs = []
    train_accuracy = []
    test_accuracy = []
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            load_ckpt_dir =  dir_path + "ckpt_"+str(i+1)+".pt"
            state = torch.load(load_ckpt_dir, map_location=device)
            epochs.append(i+1)
            train_accuracy.append(state['train_accuracy'])
            test_accuracy.append(state['test_accuracy'])
        
        print("train_acc: {}, test_acc: {}".format(train_accuracy[-1], test_accuracy[-1]))
        plt.plot(epochs, train_accuracy, color='red', label='train(pretrained)')
        plt.plot(epochs, test_accuracy, color='blue', label='test(pretrained)')

    dir_path = "ckpt/no_pretrained/resnet50/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    all_file_name = os.listdir(dir_path)
    epochs = []
    train_accuracy = []
    test_accuracy = []
    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            load_ckpt_dir =  dir_path + "ckpt_"+str(i+1)+".pt"
            state = torch.load(load_ckpt_dir, map_location=device)
            epochs.append(i+1)
            train_accuracy.append(state['train_accuracy'])
            test_accuracy.append(state['test_accuracy'])

        print("train_acc: {}, test_acc: {}".format(train_accuracy[-1], test_accuracy[-1]))
        plt.plot(epochs, train_accuracy, color='green', label='train(no pretrained)')
        plt.plot(epochs, test_accuracy, color='orange', label='test(no pretrained)')

    plt.legend()
    plt.title('Accuracy Comparison(ResNet50)') 
    plt.xlabel('epoch') # 設定 x 軸標題
    plt.ylabel('accuracy') # 設定 y 軸標題
    plt.show()


    



