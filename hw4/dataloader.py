import pandas as pd
from torch.utils import data
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
# import jpeg4py
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_name, self.label = getData(mode)
        self.root = root
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + '/' + self.mode + '/' + self.img_name[index] + '.jpeg'
        img = Image.open(path).convert('RGB')

        if self.mode == 'train':
            transform = T.Compose([
                T.ToTensor(), # [H, W, C] to [C, H, W], and convert to [0,1]
                T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop((512,512)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 從ImageNet dataset 取得這些數字的
                T.RandomRotation(degrees=20),
                T.RandomHorizontalFlip()
            ])
        else:
            transform = T.Compose([
                T.ToTensor(), # [H, W, C] to [C, H, W], and convert to [0,1]
                T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop((512,512)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        img = transform(img)
        label = int(self.label[index])

        return img, label
