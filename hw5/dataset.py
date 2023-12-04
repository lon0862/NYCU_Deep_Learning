import torch
import os
import csv
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

default_transform = transforms.Compose([
	transforms.ToTensor(),
])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.root = '{}/{}'.format(args.data_root, mode)
        self.transform = transform
        self.dirs = []
        self.seq_len = args.n_past + args.n_future
        self.seed_is_set = True ## Whether the random seed is already set or not
        for record_dir in os.listdir(self.root):
            for index_dir in os.listdir(os.path.join(self.root, record_dir)):
                self.dirs.append(os.path.join(self.root, record_dir, index_dir))
            
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return len(self.dirs)
        
    def get_seq(self, index):    
        frame_files = [file for file in os.listdir(self.dirs[index]) if ".png" in file]
        frame_files.sort(key=lambda x: int(x.split(".")[0]))

        image_seq = []
        for i in range(self.seq_len):
            fname = '{}/{}'.format(self.dirs[index], f'{i}.png')
            img = Image.open(fname)
            image_seq.append(self.transform(img))

        ## Transform to tensor of shape (12, 3, 64, 64)
        image_seq = torch.stack(image_seq)

        return image_seq
    
    def get_csv(self, index):
        with open('{}/actions.csv'.format(self.dirs[index]), newline='') as csvfile:
            rows = csv.reader(csvfile)
            actions = []
            for i, row in enumerate(rows):
                if i >= self.seq_len:
                    break
                action = [float(value) for value in row]
                actions.append(torch.tensor(action))
            
            actions = torch.stack(actions)
            
        with open('{}/endeffector_positions.csv'.format(self.dirs[index]), newline='') as csvfile:
            rows = csv.reader(csvfile)
            positions = []
            for i, row in enumerate(rows):
                if i >= self.seq_len:
                    break
                position = [float(value) for value in row]
                positions.append(torch.tensor(position))
            positions = torch.stack(positions)

        ## Concatenate 2 conditions, transform to tensor of shape (12, 7)
        condition = torch.cat((actions, positions), axis=1)

        return condition

    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq(index)
        cond =  self.get_csv(index)
        return seq, cond
