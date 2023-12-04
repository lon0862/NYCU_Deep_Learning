import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_num=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_num, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride_num > 1:
            # downsample
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_num, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = lambda x: x

        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x: torch.tensor):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.identity(x)
        out = self.relu(out)
        return out

class ResNet_18(nn.Module):
    def __init__(self):
        super(ResNet_18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64, stride_num=1), 
            ResBlock(64, 64, stride_num=1)
        )
        self.conv3 = nn.Sequential(
            ResBlock(64, 128, stride_num=2), 
            ResBlock(128, 128, stride_num=1)
        )
        self.conv4 = nn.Sequential(
            ResBlock(128, 256, stride_num=2), 
            ResBlock(256, 256, stride_num=1)
        )
        self.conv5 = nn.Sequential(
            ResBlock(256, 512, stride_num=2), 
            ResBlock(512, 512, stride_num=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)
        

    def forward(self, x: torch.tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class ResBottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride_num=1, expansion=False):
        super(ResBottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride_num, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride_num > 1 or expansion:
            # downsample
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_num, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = lambda x: x

        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x: torch.tensor):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.identity(x)
        out = self.relu(out)

        return out

class ResNet_50(nn.Module):
    def __init__(self):
        super(ResNet_50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBottleNeck(64, 64, 256, stride_num=1, expansion=True), 
            ResBottleNeck(256, 64, 256, stride_num=1, expansion=False),
            ResBottleNeck(256, 64, 256, stride_num=1, expansion=False)
        )
        self.conv3 = nn.Sequential(
            ResBottleNeck(256, 128, 512, stride_num=2, expansion=False), 
            ResBottleNeck(512, 128, 512, stride_num=1, expansion=False),
            ResBottleNeck(512, 128, 512, stride_num=1, expansion=False)
        )
        self.conv4 = nn.Sequential(
            ResBottleNeck(512, 256, 1024, stride_num=2, expansion=False), 
            ResBottleNeck(1024, 256, 1024, stride_num=1, expansion=False),
            ResBottleNeck(1024, 256, 1024, stride_num=1, expansion=False)
        )
        self.conv5 = nn.Sequential(
            ResBottleNeck(1024, 512, 2048, stride_num=2, expansion=False), 
            ResBottleNeck(2048, 512, 2048, stride_num=1, expansion=False),
            ResBottleNeck(2048, 512, 2048, stride_num=1, expansion=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 5)

    def forward(self, x: torch.tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        return x