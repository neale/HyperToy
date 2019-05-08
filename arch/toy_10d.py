import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
import utils


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.z*2)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x, training=True):
        #print ('E in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = x.view(-1, self.ze) #flatten filter size
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 2, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        #print ('E out: ', x.shape)
        return w1, w2


""" Linear (20 x 200) """
class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 100*self.nd + 100)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x, training=True):
        #print ('W1 in : ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :100*self.nd], x[:, -100:]
        w = w.view(-1, 100, self.nd)
        b = b.view(-1, 100)
        #print ('W1 out: ', x.shape)
        return w, b


""" Linear (200 x 1) """
class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 4*100 + 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x, training=True):
        #print ('W2 in : ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :100*4], x[:, -4:]
        w = w.view(-1, 4, 100)
        b = b.view(-1, 4)
        #print ('W2 out: ', x.shape)
        return w, b


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorZ'
        self.linear1 = nn.Linear(self.z, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)
        self.relu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = x.view(-1, self.z)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        # print ('Dz out: ', x.shape)
        return x
