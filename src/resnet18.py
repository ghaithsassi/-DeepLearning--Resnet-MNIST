import torch
import torch.nn as nn
import numpy as np

from basicblock import BasicBlock
from inceptionblock import InceptionBlock


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,bias=False)
        self.inception = InceptionBlock()
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2_x = nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64)
        )

        self.conv3_x = nn.Sequential(
            BasicBlock(64,128,2),
            BasicBlock(128,128)
        )

        self.conv4_x = nn.Sequential(
            BasicBlock(128,256,2),
            BasicBlock(256,256)
        )

        # self.conv5_x = nn.Sequential(
        #     BasicBlock(256,512,2),
        #     BasicBlock(512,512)
        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.FC = nn.Sequential(
            nn.Linear(in_features=256,out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64,out_features=10)
        )
        self.AC = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.inception(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        # x = self.conv5_x(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.FC(x)
        x = self.AC(x)
        return x
