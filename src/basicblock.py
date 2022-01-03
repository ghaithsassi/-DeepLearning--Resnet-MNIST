import torch
import torch.nn as nn
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample:nn.Module = None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=stride,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        if stride != 1 and downsample is None:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.downsample = downsample
    def forward(self,x):
        identity =x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out