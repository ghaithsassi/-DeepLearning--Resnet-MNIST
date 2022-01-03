import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

N = 64

#Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.FC1 = nn.Linear(784, 128)
        self.AC1 = nn.ReLU()
        self.FC2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.FC1(x)
        #x = torch.sigmoid(x)
        x = self.AC1(x)
        x = self.FC2(x)
        return x

#Define the model 
model = Net()
model = model.to(device)
print(model)

#Random input
input = torch.randn(N, 784)
input = input.to(device)
#Rondom target
target = torch.randn(N,10)
target = target.view(N, -1) # make it the same shape as output
target = target.to(device)

#Define Loss function
criterion = nn.MSELoss()
#Define the optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay= 1e-3, momentum = 0.9)

for t in range(500):
    output = model.forward(input)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if t % 100 == 99:
        print(t, loss)
