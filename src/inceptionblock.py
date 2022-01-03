import torch
import torch.nn as nn
import numpy as np

class InceptionBlock(nn.Module):
    def __init__(self):
        super(InceptionBlock,self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(1,1),bias=False)
        self.conv3x3_1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),padding=(1,1),stride=(1,1),bias=True)
        self.conv5x5_1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(5,5),padding=(2,2),bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3),stride=(1,1))
        self.conv1x1_2 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(1,1),padding=(1,1),bias=True)
    def forward(self,x):
        x0 = self.conv1x1_1(x)
        x1 = self.conv3x3_1(x)
        x2 = self.conv5x5_1(x)
        x3 = self.maxpool(x)
        x3 = self.conv1x1_2(x3)
        x = torch.cat([x0,x1,x2,x3],dim=1)
        return x