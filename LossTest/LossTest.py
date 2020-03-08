import torch
import random

import numpy as np 
import torch.nn as nn 
import torch.optim as optim

from torch.utils.data import DataLoader

# 自定义dataset类
class mydataset():
    def __init__(self, cord):
        self.cord = cord

    def __getitem__(self, index):
        x = torch.from_numpy(np.array([self.cord[index][0]], dtype=np.float32))
        label = torch.from_numpy(np.array([self.cord[index][1]], dtype=np.float32))
        return x, label

    def __len__(self):
        return len(self.cord)

# 自定义网络
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.linear1 =  nn.Linear(in_features=1, out_features=10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=10, out_features=5)
        self.linear3 = nn.Linear(in_features=5, out_features=1)


    def forward(self, input):
        x = self.linear1(input)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

# 自定义损失函数
class myloss(nn.Module):
    def __init__(self):
        super(myloss,self).__init__()
        
    def forward(self, x, y):
        # print(x, '|', y)
        loss = torch.mean((x - y)**2)
        print(type(loss))
        return loss

# 训练
def train(epoch):
    lossSum, numSUm = 0, 0
    for batch, (x, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = net(x)
        loss = lossFc(output, label)
        
        lossSum += loss.data
        numSUm += len(x)

        loss.backward()
        optimizer.step()
        # print("Epoch: {} | Batch: {} | Loss: {:.3f}".format(epoch, batch, loss))
    print("--Epoch: {} | AvgLoss: {:.3f}".format(epoch, lossSum/numSUm))

# 数据准备
Residual = lambda a: random.randint(-int(a), int(a))/9.8
x = [i/10 for i in range(1, 100, 1)]+[i/5 for i in range(100, 200, 1)]
y = [i+Residual(i) for i in x]
cord = list(map(lambda x, y: (x, y), x, y))
# cord = list(map(lambda x, y: (x, y), x, y))+[(1., 2.), (30., 50.), (10., 20.)]

# 创建dataset与dataloader
train_dataset = mydataset(cord)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=3, shuffle=True, num_workers=1)

# 准备模型网络,损失,优化器
net = mynet()
# lossFc = nn.L1Loss()
lossFc = myloss()
optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-5)

if __name__ == "__main__":
    for epoch in range(1, 10):
        train(epoch)