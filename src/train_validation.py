#!/usr/bin/env python
# coding: utf-8



# load packages
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import math

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
# from torch2trt import torch2trt



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# 40 features
def prepare_x(data):
    df1 = data[:40, :].T
    #print('df1 shape: {}'.format(df1.shape))
    return np.array(df1)

# 5 horizon k
def get_label(data):
    lob = data[-5:, :].T
    #print('lob shape: {}'.format(lob.shape))
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape     
    #print('N, D = {}'.format(X.shape))
    df = np.array(X)     
    #print('df: {}'.format(df.shape))

    dY = np.array(Y)     
    #print('dY: {}'.format(dY.shape))

    dataY = dY[T - 1:N]  
    #print('dataY shape: {}'.format(dataY.shape))

    dataX = np.zeros((N - T + 1, T, D))  
    #print('dataX shape: {}'.format(dataX.shape))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

# unused
def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y




class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:,self.k] - 1
        self.length = len(x)
        #print('len(x): {}'.format(self.length))

        x = torch.from_numpy(x)         
        #print('x before unsqueeze: {}'.format(x.shape))
        self.x = torch.unsqueeze(x, 1)  
        #print('x after unsqueeze: {}'.format(self.x.shape))
        self.y = torch.from_numpy(y)    
        #print('y: {}'.format(self.y.shape))

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]





''' 
please change the data_path to your local path
'''

dec_data = np.loadtxt('Train_Dst_NoAuction_ZScore_CF_7.txt')
print(dec_data.shape)

dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.9))]
dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.9)):]

dec_test1 = np.loadtxt('Test_Dst_NoAuction_ZScore_CF_7.txt')
dec_test2 = np.loadtxt('Test_Dst_NoAuction_ZScore_CF_8.txt')
dec_test3 = np.loadtxt('Test_Dst_NoAuction_ZScore_CF_9.txt')
dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

print(dec_train.shape, dec_val.shape, dec_test.shape)



batch_size = 32

dataset_train = Dataset(data=dec_train, k=1, num_classes=3, T=10)
dataset_val = Dataset(data=dec_val, k=1, num_classes=3, T=10)
dataset_test = Dataset(data=dec_test, k=1, num_classes=3, T=10)

print(dataset_train.x.shape, dataset_train.y.shape)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)



## ResNeXt block


class ResNeXt_Block(nn.Module):
    def __init__(self, in_chnls, out_chnls, kernel_size, cardinality, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = out_chnls//2
        
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(in_chnls),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=in_chnls, out_channels=self.group_chnls, kernel_size=(1,1), stride=(1,1)),
        )
        self.groupconv = nn.Sequential(
            nn.InstanceNorm2d(self.group_chnls),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=self.group_chnls, out_channels=self.group_chnls, kernel_size=kernel_size,
                      stride=stride, groups=cardinality),
        )
        self.conv3 = nn.Sequential(
            nn.InstanceNorm2d(self.group_chnls),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=self.group_chnls, out_channels=out_chnls, kernel_size=(1,1), stride=(1,1)),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.groupconv(x)
        x = self.conv3(x)
        return x


## MobileNet


class MobileNet_block(nn.Module):
    def __init__(self, channel):
        super(MobileNet_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), padding=(1,1), groups=channel),
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1,1), stride=(1,1)),
            nn.InstanceNorm2d(channel),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class MobileNet_block2(nn.Module):
    def __init__(self, channel):
        super(MobileNet_block2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3,1), padding=(1,0), groups=channel),
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1,1), stride=(1,1)),
            nn.InstanceNorm2d(channel),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


## SENet


class SELayer_tanh(nn.Module):
    def __init__(self, channel, reduction=32):
        super(SELayer_tanh, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            #nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# # TCN temporal block



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.size(dim=2) - self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.leakyrelu1, self.dropout1,
                                 self.conv2, self.chomp2, self.leakyrelu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.leakyrelu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# # main block



class deeplob(nn.Module):
    def __init__(self, num_classes=3):
        super(deeplob, self).__init__()
        
        self.ResNeXt_Block1 = ResNeXt_Block(in_chnls=1, out_chnls=32, kernel_size=(1,2), cardinality=16, stride=(1,2))
        self.MobileNet1 = MobileNet_block(channel=32)
        self.MobileNet2 = MobileNet_block(channel=32)
        
        self.se_1 = SELayer_tanh(channel=32)
        
        self._downsample_1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=1, stride=(1,2), bias=False)
                                           , nn.InstanceNorm2d(32))
        
        
        self.ResNeXt_Block2 = ResNeXt_Block(in_chnls=32, out_chnls=32, kernel_size=(1,2), cardinality=16, stride=(1,2))
        self.MobileNet3 = MobileNet_block(channel=32)
        self.MobileNet4 = MobileNet_block(channel=32)
        
        self._downsample_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, stride=(1,2), bias=False)
                                           , nn.InstanceNorm2d(32))
        
        
        self.ResNeXt_Block3 = ResNeXt_Block(in_chnls=32, out_chnls=32, kernel_size=(1,10), cardinality=16, stride=(1,10))
        self.MobileNet5 = MobileNet_block2(channel=32)
        self.MobileNet6 = MobileNet_block2(channel=32)
        
        self._downsample_3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, stride=(1,10), bias=False)
                                           , nn.InstanceNorm2d(32))
        
        
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        
        self.tcn = TemporalConvNet(num_inputs=32, num_channels=[32, 32], kernel_size=4, dropout=0.2)
        
        # fully connected
        self.fc1 = nn.Linear(32, num_classes)

    
    
    def forward(self, x):
        out = self.ResNeXt_Block1(x)
        out = self.MobileNet1(out)
        out = self.leakyrelu(out)
        out = self.MobileNet2(out)
        out = self.se_1(out)
        identity = self._downsample_1(x)
        out += identity
        out = self.leakyrelu(out)
        
        identity = out
        out = self.ResNeXt_Block2(out)
        out = self.MobileNet3(out)
        out = self.leakyrelu(out)
        out = self.MobileNet4(out)
        identity = self._downsample_2(identity)
        out += identity
        out = self.leakyrelu(out)
        
        identity = out
        out = self.ResNeXt_Block3(out)
        out = self.MobileNet5(out)
        out = self.leakyrelu(out)
        out = self.MobileNet6(out)
        identity = self._downsample_3(identity)
        out += identity
        out = self.leakyrelu(out)
        
        out = out.view(-1, out.shape[1], out.shape[2])
        out = self.tcn(out)
        out = out[:, :, out.size(2) - 1]
        
        out = self.fc1(out)
        
        return out




model = deeplob()
model.to(device)




summary(model, (1, 1, 10, 40))




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)




# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0
    not_improved = 0
    patience = 100
    training_time = 0

    for it in tqdm(range(epochs)):
        
        model.train()
        training_time += 1
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # print("inputs.shape:", inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            outputs = model(inputs)
            # print("done getting model output")
            # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        scheduler.step()
        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading
    
        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)      
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        if test_loss < best_test_loss:
            torch.save(model, './best_val_model_pytorch')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')
            not_improved = 0
        else :
            not_improved += 1
            print(f'early stopping in {not_improved}/{patience}')

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f},           Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch+1}')
        
        if not_improved >= patience and training_time >= 10:
            print('Early stopping at epoch: {}'.format(it+1))
            print('Finish training')
            return train_losses, test_losses

    return train_losses, test_losses



train_losses, val_losses = batch_gd(model, criterion, optimizer, 
                                    train_loader, val_loader, epochs=72)




plt.figure(figsize=(15,6))
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='validation loss')
plt.legend()
