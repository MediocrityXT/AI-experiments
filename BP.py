import numpy as np
import pandas as pd
import torch
import time
from torch._C import dtype
from torch.tensor import Tensor
import torch.utils.data as d
import torch.nn as nn
from torch.utils.data import sampler
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

def getTrainDataLoader():
    pd_data = pd.read_table('AI experiments/data/Iris-train.txt',names=['a1','a2','a3','a4','l'],sep=' ',usecols=[0,1,2,3,4])
    x=np.array(pd_data, dtype = np.float32)
    trainset=torch.from_numpy(x)
    trainset=d.TensorDataset(trainset[:,:4],trainset[:,4])
    loader = d.DataLoader(
        dataset=trainset,      # format is torch TensorDataset 
        batch_size=1,      # mini batch size
        shuffle=False,               # random shuffle for training 随机洗牌训练
        #num_workers=2,              # subprocesses for loading data
    )
    return loader

def getTestDataLoader():
    pd_data = pd.read_table('AI experiments/data/Iris-test.txt',names=['a1','a2','a3','a4','l'],sep=' ',usecols=[0,1,2,3,4])
    x=np.array(pd_data, dtype = np.float32)
    testset=torch.from_numpy(x)
    testset=d.TensorDataset(testset[:,:4],testset[:,4])
    loader = d.DataLoader(
        dataset=testset,      # format is torch TensorDataset 
        batch_size=1,      # mini batch size
        shuffle=False,               # random shuffle for training 随机洗牌训练
        #num_workers=2,              # subprocesses for loading data
    )
    return loader

def getLabelProb(label):
    res = torch.zeros(1,3)
    cIndex = int(label.numpy()[0])
    res[0,cIndex] = 1 
    res.requires_grad_()
    return res

class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__() #init nn.Module
        
        self.f1=nn.Linear(4,10,True) #fully connect = linear transform
        self.f2=nn.Linear(10,3,True)
        
    def forward(self, x):
        #sigmoid=nn.Sigmoid()
        x = torch.sigmoid(self.f1(x))
        x = torch.sigmoid(self.f2(x))
        return x
        
'''
print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
def train():
    criterion = nn.MSELoss()
    optimizer = optim.SGD(Net.parameters(), lr=0.1) 
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1, last_epoch=-1)
    loader = getTrainDataLoader()
    for epoch in range(500):  
    #  loop over the dataset multiple times
        runningLoss=0.0
        #scheduler.step()
        for i, (inputs,labels) in enumerate(loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            #inputs = inputs.to(device)
            
            # forward + backward + optimize
            label=getLabelProb(labels)
            #label = label.to(device)
            #print('[%d, %d] :' % (epoch , i),'| input : ',inputs, ' labelPossibility : ', label)
            outputs = Net(inputs)
            outputs.requires_grad_()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            
            if i == 74:
                # print statistics
                #print(Net.f1.weight)
                print('[%d, %d] loss: %.3f' % (epoch , i , runningLoss))
            
    print('Finished Training')

Net= BPNet()
#Net.to(device)
start = time.process_time()
train()
elapse=time.process_time() - start
print("timeUsed: "+str(elapse))
testloader=getTestDataLoader()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = Net(inputs)
        total += 1
        res = outputs.max(axis=-1)[1]
        res = res.float()
        if(res == labels):
            correct += 1

print('Accuracy : ' , correct / total)
input()
