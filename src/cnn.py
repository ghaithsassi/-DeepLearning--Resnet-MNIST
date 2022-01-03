import torch
import torch.nn as nn
import numpy as np


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        self.conv1= nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3),padding=(1,1))
        self.AC1 =nn.ReLU(inplace=True)
        self.conv2= nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=(1,1))
        self.AC2 = nn.ReLU(inplace=True)
        self.maxpool =nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=6272,out_features=10)
        self.AC3 = nn.Softmax(dim=-1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.AC1(x)
        x = self.conv2(x)
        x = self.AC2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.AC3(x)
        return x