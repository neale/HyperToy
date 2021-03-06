import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
import utils


class MixerS(nn.Module):
    def __init__(self, args):
        super(MixerS, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoderz'
        self.linear1 = nn.Linear(self.ze, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.z*4)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(-1, self.ze) #flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 4, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        w4 = x[:, 3]
        #print ('E out: ', x.shape)
        return (w1, w2, w3, w4)


class GeneratorE1(nn.Module):
    def __init__(self, args):
        super(GeneratorE1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE1'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 100*2 + 100)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, training=True):
        #print ('W2 in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :100*2], x[:, -100:]
        w = w.view(-1, 100, 2)
        b = b.view(-1, 100)
        #print ('W2 out: ', x.shape)
        return (w, b)

class GeneratorE2(nn.Module):
    def __init__(self, args):
        super(GeneratorE2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE2'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 100*1 + 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, training=True):
        #print ('W2 in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :1*100], x[:, -1:]
        w = w.view(-1, 1, 100)
        b = b.view(-1, 1)
        #print ('W2 out: ', x.shape)
        return (w, b)

class GeneratorD1(nn.Module):
    def __init__(self, args):
        super(GeneratorD1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD1'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1*100 + 100)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, training=True):
        #print ('W2 in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :1*100], x[:, -100:]
        w = w.view(-1, 100, 1)
        b = b.view(-1, 100)
        #print ('W2 out: ', x.shape)
        return (w, b)


class GeneratorD2(nn.Module):
    def __init__(self, args):
        super(GeneratorD2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD1'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 2*100 + 2)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, training=True):
        #print ('W2 in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :100*2], x[:, -2:]
        w = w.view(-1, 2, 100)
        b = b.view(-1, 2)
        #print ('W2 out: ', x.shape)
        return (w, b)


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
