import torch
import torch.nn as nn
import numpy as np

from src.basicblock import BasicBlock
from src.inceptionblock import InceptionBlock


class Resnetv2(nn.Module):
    def __init__(self):
        super(Resnetv2, self).__init__()

        self.inception = InceptionBlock()
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_x = nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64)
        )
        self.conv3_x = nn.Sequential(
            BasicBlock(64,128,2),
            BasicBlock(128,128)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.FC = nn.Sequential(
            nn.Linear(in_features=6272,out_features=10)
        )
        self.AC = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.inception(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.FC(x)
        x = self.AC(x)
        return x
