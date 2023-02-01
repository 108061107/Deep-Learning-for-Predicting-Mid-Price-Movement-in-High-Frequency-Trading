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
from torch2trt import torch2trt



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# 40 features
def prepare_x(data):
    df1 = data[:40, :].T  
    return np.array(df1)

# 5 horizon k
def get_label(data):
    lob = data[-5:, :].T  
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape     
    df = np.array(X)     

    dY = np.array(Y)     

    dataY = dY[T - 1:N] 

    dataX = np.zeros((N - T + 1, T, D))  
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
        self.length = len(x)  # 203701

        x = torch.from_numpy(x)        
        self.x = torch.unsqueeze(x, 1)  
        self.y = torch.from_numpy(y)    

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]




# please change the data_path to your local path
# data_path = '/nfs/home/zihaoz/limit_order_book/data'

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


## TCN temporal block
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


## main block
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



model = torch.load('best_val_model_pytorch')

all_targets = []
all_predictions = []

for inputs, targets in test_loader:
    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    outputs = model(inputs)
    
    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)

    all_targets.append(targets.cpu().numpy())
    all_predictions.append(predictions.cpu().numpy())

all_targets = np.concatenate(all_targets)    # concatenate each batch
all_predictions = np.concatenate(all_predictions)   # concatenate each batch



print(f"Accuracy_score: {accuracy_score(all_targets, all_predictions):.4f}")  # return the fraction of correctly classified samples
print(f"Precision: {precision_score(all_targets, all_predictions, average='weighted'):.4f}")
print(f"Recall: {recall_score(all_targets, all_predictions, average='weighted'):.4f}")
print(f"F1 score: {f1_score(all_targets, all_predictions, average='weighted'):.4f}")
print(classification_report(all_targets, all_predictions, digits=4))    # Text summary of the precision, recall, F1 score for each class





tmp_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=True)
dt = 0
test_time = 0

for inputs, targets in tmp_loader:
    start_time = time.time()
    
    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    #outputs = model(inputs)
    outputs = model(inputs)

    # Get prediction
    # torch.max returns both max(value) and argmax(index)
    _, predictions = torch.max(outputs, 1)
    
    torch.cuda.synchronize()
    end_time = time.time()
    dt += end_time - start_time
    
    test_time += 1
    if test_time >= 100:
        break


avg_latency = dt / 100
print(f"Average latency: {avg_latency}")
