import torch
import torch.nn as nn
import numpy as np

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

import optuna


log_info_console = False
gpu = torch.cuda.is_available() #set to False to train on CPU
device = torch.device("cuda:0" if gpu else "cpu")
print(device)

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

#fixed hyperparams
epochs = 16


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


def objective(trial):
    
    batch_size = trial.suggest_categorical('batch_size',[5,64,200])
    learning_rate =  trial.suggest_categorical('learning_rate',[1.0,0.1,0.01,0.001])
    weight_decay  = 1e-3#trial.suggest_categorical('weight_decay',[0.1,0.01,0.001])
    
    optimizer_name = torch.optim.SGD #trial.suggest_categorical('optimizer_name',[torch.optim.Adam,torch.optim.SGD])
    model_name = Resnet18 #trial.suggest_categorical('model_name',[Cnn,Resnet18,Resnetv2])


    
    train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=mnist_testset,batch_size=batch_size,shuffle=False)

    model = model_name()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_name(model.parameters(),lr=learning_rate, weight_decay= weight_decay)

    for epoch in range(epochs):
        acc_list = []
        for batch_idx,(input,target) in enumerate(train_loader):
            optimizer.zero_grad()
            if gpu:
                input,target = input.to(device),target.to(device)
            output = model.forward(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            ###
            running_loss = loss.item()
            _,pred_label =torch.max(output.data,1)
            correct_label = (pred_label==target.data).sum()
            acc = correct_label*1.0/batch_size
            if log_info_console and ( (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader)):
                print("train epoch:{}, batch:{}, loss:{}, accuracy:{}".format(epoch,batch_idx,running_loss,acc))
    test_acc = []
    with torch.no_grad():
        for batch_idx,(input,target) in enumerate(test_loader):
            if gpu:
                input,target = input.to(device),target.to(device)
            output = model.forward(input)
            loss = criterion(output, target)
            _,pred_label =torch.max(output.data,1)
            correct_label = (pred_label==target.data).sum()
            acc = correct_label*1.0/batch_size
            test_acc.append(acc)
            if log_info_console and ((batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader)):
                print("test epoch:{}, batch:{}, loss:{}, accuracy:{}".format(epoch,batch_idx,running_loss,acc))
    test_accuracy = torch.FloatTensor(test_acc).mean()
    return float(test_accuracy)



study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=16)


print(len(study.trials))
print(study.best_params)