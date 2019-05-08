import os
import sys
import time
import torch
import natsort
import datagen
import argparse
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from scipy.misc import imsave
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
import torch.distributions.uniform as U


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D

def create_full(shape):
    mean = torch.zeros(shape)
    cov = torch.ones((shape, shape))
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad = grad
    return z


def create_uniform():
    x = U.Uniform(torch.tensor(0.0), torch.tensor(1.0))
    return x


def sample_uniform(D, shape):
    x = D.sample((shape)).cuda()
    x.requires_grad_(True)
    return x


def sample_z_like(shape, scale=1., grad=True):
    return torch.randn(shape, requires_grad=True).cuda()


def batch_zero_grad(nets):
    for module in nets:
        module.zero_grad()


def save_model(args, model, optim):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'best_acc': args.best_acc,
        'best_loss': args.best_loss
        }, path)


def load_model(args, model, optim):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    acc = ckpt['best_acc']
    loss = ckpt['best_loss']
    return model, optim, (acc, loss)


def get_net_only(model):
    net_dict = {
            'state_dict': model.state_dict(),
    }
    return net_dict


def load_net_only(model, d):
    model.load_state_dict(d['state_dict'])
    return model


def save_hypernet_layer(args, models, acc):
    netE, netD, netG = models
    hypernet_dict = {
            'E': get_net_only(netE),
            'D': get_net_only(netD),
            'G': get_net_only(netG),
            }
    path = 'saved_models/mnist/hyperlayer_{}_{}.pt'.format(args.exp, acc)
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


def save_hypernet_mnist(args, models, acc):
    netE, netD, W1, W2, W3 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'D':  get_net_only(netD),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            }
    path = 'saved_models/mnist/hypermnist_{}_{}.pt'.format(args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


def save_hypernet_lrg(args, models, acc):
    netE, netD, W1, W2, W3, W4 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'D':  get_net_only(netD),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            'W4': get_net_only(W4),
            }
    path = 'saved_models/mnist/hypermnist_lrg_{}_{}.pt'.format(args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


def save_hypernet_cifar(args, models, acc):
    netE, netD, W1, W2, W3, W4, W5 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            'W4': get_net_only(W4),
            'W5': get_net_only(W5),
            'D': get_net_only(netD),
            }
    path = 'saved_models/cifar/hypercifar_{}_{}.pt'.format(args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


def save_hypernet_regression(args, models, mse):
    netE, W1, W2 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            }
    path = 'exp_models/hypertoy{}_{}.pt'.format(args.exp, mse)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


""" hard coded for mnist experiment dont use generally """
def load_hypernet_mnist(args, path):
    #import models.models_mnist_nobias as models
    #import models.models_mnist_small as models
    #import arch.models_mnist_noadds as models
    import arch.models_mnist_lrg as models
    netE = models.Encoderz(args).cuda()
    netD = models.DiscriminatorQz(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    W3 = models.GeneratorW3(args).cuda()
    W4 = models.GeneratorW4(args).cuda()
    print ('loading hypernet from {}'.format(path))
    d = torch.load(path)
    netE = load_net_only(netE, d['E'])
    netD = load_net_only(netD, d['D'])
    W1 = load_net_only(W1, d['W1'])
    W2 = load_net_only(W2, d['W2'])
    W3 = load_net_only(W3, d['W3'])
    W4 = load_net_only(W4, d['W4'])
    return (netE, netD, W1, W2, W3, W4)


def load_hypernet_cifar(args, path):
    import arch.models_cifar_5k as hyper
    netE = hyper.Encoder(args).cuda()
    W1 = hyper.GeneratorW1(args).cuda()
    W2 = hyper.GeneratorW2(args).cuda()
    W3 = hyper.GeneratorW3(args).cuda()
    W4 = hyper.GeneratorW4(args).cuda()
    W5 = hyper.GeneratorW5(args).cuda()
    print ('loading hypernet from {}'.format(path))
    d = torch.load(path)
    netE = load_net_only(netE, d['E'])
    W1 = load_net_only(W1, d['W1'])
    W2 = load_net_only(W2, d['W2'])
    W3 = load_net_only(W3, d['W3'])
    W4 = load_net_only(W4, d['W4'])
    W5 = load_net_only(W5, d['W5'])
    return (netE, W1, W2, W3, W4, W5)


def sample_hypernet_mnist(args ,hypernet, num):
    netE, W1, W2, W3 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    return l1, l2, l3, codes


def sample_hypernet_cifar(args, hypernet, num):
    netE, W1, W2, W3, W4, W5 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    l4 = W4(codes[3])
    l5 = W5(codes[4])
    return l1, l2, l3, l4, l5, codes


def weights_to_clf(weights, model, names):
    state = model.state_dict()
    layers = zip(names, weights)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        model.load_state_dict(state)
    return model
