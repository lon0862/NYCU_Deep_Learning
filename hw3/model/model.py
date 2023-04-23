import torch
import torch.nn as nn
import numpy as np

class EEGNet(nn.Module):
    def __init__(self, act_model):
        super(EEGNet, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16))

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            act_model,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.25)
            )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            act_model,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(0.25)
            )
        
        self.classify = nn.Sequential(
            nn.Linear(736, 2)
        )

    def forward(self, x: torch.tensor): 
        pred = self.firstConv(x)
        pred = self.depthwiseConv(pred)
        pred = self.separableConv(pred)
        pred = pred.view(-1, 736)
        pred = self.classify(pred)

        return pred

class DeepConvNet(nn.Module):
    def __init__(self, act_model):
        super(DeepConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25),
            act_model,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50),
            act_model,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5))

        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100),
            act_model,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5))

        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200),
            act_model,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5))

        self.fc = nn.Linear(8600, 2)

    def forward(self, x: torch.tensor): 
        pred = self.conv1(x)
        pred = self.conv2(pred)
        pred = self.conv3(pred)
        pred = self.conv4(pred)
        pred = pred.view(-1, 8600)
        pred = self.fc(pred)

        return pred