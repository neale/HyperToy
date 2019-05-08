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
        self.linear1 = nn.Linear(self.ze, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, self.z*2)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
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

class Encoder1layer(nn.Module):
    def __init__(self, args):
        super(Encoder1layer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 100)
        self.linear2 = nn.Linear(100, self.z*2)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x, training=True):
        #print ('E in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = x.view(-1, self.ze) #flatten filter size
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 2, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        #print ('E out: ', x.shape)
        return w1, w2

class EncoderNoBN(nn.Module):
    def __init__(self, args):
        super(EncoderNoBN, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.z*2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, training=True):
        #print ('E in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = x.view(-1, self.ze) #flatten filter size
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = x.view(-1, 2, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        #print ('E out: ', x.shape)
        return w1, w2


class EncoderHour(nn.Module):
    def __init__(self, args):
        super(EncoderHour, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, self.z*2)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, training=True):
        #print ('E in: ', x.shape)
        if training:
            x = x + torch.randn_like(x)
        x = x.view(-1, self.ze) #flatten filter size
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.relu(self.bn4(self.linear4(x)))
        x = self.linear5(x)
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
        self.linear3 = nn.Linear(128, 2*512 + 512)
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
        w, b = x[:, :512*2], x[:, -512:]
        w = w.view(-1, 512, 2)
        b = b.view(-1, 512)
        #print ('W1 out: ', x.shape)
        return w, b

""" Linear (512 x 512) """
class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 512*512 + 512)
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
        w, b = x[:, :512*512], x[:, -512:]
        w = w.view(-1, 512, 512)
        b = b.view(-1, 512)
        #print ('W1 out: ', x.shape)
        return w, b


""" Linear (512 x 512) """
class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 512*512 + 512)
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
        w, b = x[:, :512*512], x[:, -512:]
        w = w.view(-1, 512, 512)
        b = b.view(-1, 512)
        #print ('W1 out: ', x.shape)
        return w, b


""" Linear (512 x 512) """
class GeneratorW4(nn.Module):
    def __init__(self, args):
        super(GeneratorW4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW4'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 512*512 + 512)
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
        w, b = x[:, :512*512], x[:, -512:]
        w = w.view(-1, 512, 512)
        b = b.view(-1, 512)
        #print ('W1 out: ', x.shape)
        return w, b


""" Linear (200 x 1) """
class GeneratorW5(nn.Module):
    def __init__(self, args):
        super(GeneratorW5, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW5'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 4*512 + 4)
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
        w, b = x[:, :512*4], x[:, -4:]
        w = w.view(-1, 4, 512)
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
