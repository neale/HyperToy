import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import pprint
import argparse
import numpy as np
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

import args as arg
import netdef, datagen
import ops, mnist, test, utils
import stats_mnist as stats
import arch.mnist_clf as mnist_clf
import arch.models_toy as models
import ensemble
from torch.distributions.multivariate_normal import MultivariateNormal

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=32, type=int, help='latent space width')
    parser.add_argument('--ze', default=32, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--dataset', default='gaussian', type=str)
    parser.add_argument('--beta', default=10, type=int)
    parser.add_argument('--l', default=10, type=int)
    parser.add_argument('--pretrain_e', default=True, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    args = parser.parse_args()
    return args


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(2, 100)
        self.linear3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear3(x)


def train_nn(args, Z, data, target, p=False):
    """ calc classifier loss on target architecture """
    data, target = data.cuda(), target.cuda()
    target = target.view(-1)
    w1, b1 = Z[0]
    w2, b2 = Z[1]
    x = F.relu(F.linear(data, w1, b1))
    x = F.linear(x, w2, b2)
    loss = F.cross_entropy(x, target)
    pred = x.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    if p:
        print ('target', target, 'pred', pred.view(-1))
       
    return loss, correct


def plot_data(x, y, title):
    plt.close('all')
    datas = [[], [], [], []]
    for (data, target) in zip(x, y):
        datas[target].append(np.array(data))
    plt.scatter(*zip(*datas[0]), alpha=.5, linewidth=.1, edgecolor='k', label='c1')
    plt.scatter(*zip(*datas[1]), alpha=.5, linewidth=.1, edgecolor='k', label='c2')
    plt.scatter(*zip(*datas[2]), alpha=.5, linewidth=.1, edgecolor='k', label='c3')
    plt.scatter(*zip(*datas[3]), alpha=.5, linewidth=.1, edgecolor='k', label='c4')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    #plt.legend(loc='best')
    plt.savefig('figures/{}'.format(title))


def plot_data_entropy(x, y, ents, title):
    plt.close('all')
    ents = np.array(ents)
    #prob = np.array(prob)
    ents = 1 - ((ents - ents.min()) / (ents.max() - ents.min()))
    def to_color(y):
        if y == 0: return 'b'
        if y == 1: return 'y'
        if y == 2: return 'g'
        if y == 3: return 'r'

    for (data, target, ent) in zip(x, y, ents):
        plt.scatter(*data, c=to_color(target), alpha=ent)
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.title('HyperGAN')
    plt.savefig('figures/mmd_ent/ood_corse/{}'.format(title))


def plot(netE, W1, W2, iter, net=None):
    points, targets, ents, probs = [], [], [], []
    for x1 in np.linspace(-1000, 1000, 100):
        for x2 in np.linspace(-1000, 1000, 100):
            z = torch.randn(args.batch_size, args.ze).cuda()
            codes = netE(z)
            preds = []
            l1_w, l1_b = W1(codes[0], training=False)
            l2_w, l2_b = W2(codes[1], training=False)
            clf_loss, acc = 0, 0
            for (h1_w, h1_b, h2_w, h2_b) in zip(l1_w, l1_b, l2_w, l2_b):
                data = torch.tensor([x1, x2]).cuda()
                if net is not None:
                    x = net(data)
                else:
                    h1 = (h1_w, h1_b)
                    h2 = (h2_w, h2_b)
                    x = F.relu(F.linear(data, h1_w, bias=h1_b))
                    x = F.linear(x, h2_w, bias=h2_b)
                preds.append(x)
            points.append((x1, x2))
            y = torch.stack(preds).mean(0).view(1, 4)
            targets.append(F.softmax(y, dim=1).max(1, keepdim=True)[1].item())
            ents.append(entropy(F.softmax(torch.stack(preds), dim=1).mean(0).cpu().numpy().T))
            #probs.append(F.softmax(torch.stack(preds),dim=1).mean(0).max().item())
    plot_data_entropy(points, targets, ents, 'gaussian_{}'.format(iter))
    

def perm_data(x, y):
    perm = torch.randperm(len(x))
    x_perm = x[perm, :]
    y_perm = y[perm]
    return x_perm.cuda(), y_perm.cuda()


def create_data(args):
    dist1 = MultivariateNormal(torch.tensor([4.2, 6.2]), torch.eye(2)*.09)
    dist2 = MultivariateNormal(torch.tensor([4.8, 5.]), torch.eye(2)*0.025)
    dist3 = MultivariateNormal(torch.tensor([6., 6.0]), torch.eye(2)*.09)
    dist4 = MultivariateNormal(torch.tensor([6., 3.9]), torch.eye(2)*0.08)
    p1 = dist1.sample((25,))
    p2 = dist2.sample((25,))
    p3 = dist3.sample((25,))
    p4 = dist4.sample((25,))
    x = torch.stack([p1, p2, p3, p4]).view(-1, 2).cuda()
    y_base = torch.ones(25)
    y = torch.stack([y_base*0, y_base, y_base*2, y_base*3]).long().view(-1).cuda()
    return x, y


def mmd(args, x, y):
    x = x.unsqueeze(1).unsqueeze(3)
    y = y.unsqueeze(1).unsqueeze(3)
    x = x.view(x.size(0), x.size(2) * x.size(3))
    y = y.view(y.size(0), y.size(2) * y.size(3))
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    K = torch.exp(- 1 * (rx.t() + rx - 2*xx))
    L = torch.exp(- 1 * (ry.t() + ry - 2*yy))
    P = torch.exp(- 1 * (rx.t() + ry - 2*zz))
    beta = (1./(args.batch_size*(args.batch_size-1)))
    gamma = (2./(args.batch_size**2)) 
    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)


def test_far_data(netE, W1, W2, net=None):
    dist1 = MultivariateNormal(torch.tensor([4.2, 6.2]), torch.eye(2)*.001)
    dist2 = MultivariateNormal(torch.tensor([4.8, 5.]), torch.eye(2)*.001)
    dist3 = MultivariateNormal(torch.tensor([6., 6.]), torch.eye(2)*.001)
    dist4 = MultivariateNormal(torch.tensor([6., 3.9]), torch.eye(2)*.001)
    dist5 = MultivariateNormal(torch.tensor([0., 0.]), torch.eye(2)*.001)
    p1 = dist1.sample((10,)); p2 = dist2.sample((10,)); p3 = dist3.sample((10,))
    p4 = dist4.sample((10,)); p5 = dist5.sample((10,))
    for p in torch.stack([p1, p2, p3, p4, p5]):
        data = p.view(-1, 2).cuda()
        preds = []
        if net:
            x = net(data).view(-1, 4)
            ent = entropy(F.softmax(x, dim=1).cpu().numpy().T).mean()
        else:
            codes = netE(torch.randn(args.batch_size, args.ze).cuda())
            l1_w, l1_b = W1(codes[0], training=False)
            l2_w, l2_b = W2(codes[1], training=False)
            for (h1_w, h1_b, h2_w, h2_b) in zip(l1_w, l1_b, l2_w, l2_b):
                h1 = (h1_w, h1_b)
                h2 = (h2_w, h2_b)
                x = F.relu(F.linear(data, h1_w, bias=h1_b))
                x = F.linear(x, h2_w, bias=h2_b)
                preds.append(x)
                ent = entropy(F.softmax(torch.stack(preds), dim=1).mean(0).cpu().numpy().T)
                x = F.softmax(torch.stack(preds), dim=1).mean(0)
            print (ent.shape)
            print (x)
        print (p.mean().int().item(), 'uncertainty: ', ent)
    

def test_point(args, netE, W1, W2, x):
    data = torch.tensor([*x]).cuda()
    preds = []
    codes = netE(torch.randn(100, args.ze).cuda())
    l1_w, l1_b = W1(codes[0], training=False)
    l2_w, l2_b = W2(codes[1], training=False)
    for (h1_w, h1_b, h2_w, h2_b) in zip(l1_w, l1_b, l2_w, l2_b):
        h1 = (h1_w, h1_b)
        h2 = (h2_w, h2_b)
        x = F.relu(F.linear(data, h1_w, bias=h1_b))
        x = F.linear(x, h2_w, bias=h2_b)
        preds.append(x)
        #ent = entropy(F.softmax(torch.stack(preds), dim=1).mean(0).cpu().numpy().T)
        #x = F.softmax(torch.stack(preds), dim=1).mean(0)
    #print (ent.shape)
    #print (x)
    #print (p.mean().int().item(), 'uncertainty: ', ent)
    return preds
    

def train():
    args = load_args()
    netE = models.Encoder(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    print (netE, W1, W2)

    optimE = optim.Adam(netE.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=5e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=5e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=5e-4)
    
    best_test_acc, best_clf_acc, best_test_loss, = 0., 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc
    args.best_clf_loss, args.best_clf_acc = np.inf, 0

    print ('==> Creating 4 Gaussians')
    data, targets = create_data(args)
    one = torch.tensor(1.).cuda()
    mone = one * -1
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.pretrain_e is True:
        for j in range(100):
            x = torch.randn(e_batch_size, args.ze).cuda()
            qz = torch.randn(e_batch_size, args.z*2).cuda()
            codes = torch.stack(netE(x)).view(-1, args.z*2)
            mean_loss, cov_loss = ops.pretrain_loss(codes, qz)
            loss = mean_loss + cov_loss
            loss.backward()
            optimE.step()
            netE.zero_grad()
            print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
                j, mean_loss.item(), cov_loss.item()))
            final = loss.item()
            if loss.item() < 0.1:
                print ('Finished Pretraining Encoder')
                break

    """ wtf is going on here """
    net = NN().cuda()
    optimNN = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(200):
        data, targets = perm_data(data, targets)
        out = net(data)
        loss = F.cross_entropy(out, targets)
        pred = out.data.max(1, keepdim=True)[1]
        cor = pred.eq(targets.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimNN.step()
        optimNN.zero_grad()
    print (cor)
    #with torch.no_grad():
        #test_far_data(netE, W1, W2)
        #plot(netE, W1, W2, 0, net)
        #sys.exit(0)

    print ('==> Begin Training')
    for epoch in range(args.epochs):
        data, targets = perm_data(data, targets)
        # data, targets = create_data(args)
        z = torch.randn(args.batch_size, args.ze).cuda()
        ze = torch.randn(args.batch_size, args.z).cuda()
        qz = torch.randn(args.batch_size, args.z*2).cuda()
        codes = netE(z)
        
        z11 = torch.randn(args.batch_size, args.z).cuda()
        z12 = torch.randn(args.batch_size, args.z).cuda()
        z21 = torch.randn(args.batch_size, args.z).cuda()
        z22 = torch.randn(args.batch_size, args.z).cuda()
        latents11, latents12 = netE(z11)
        latents21, latents22 = netE(z21)
        dz = mmd(args, z11, z21)
        dq = mmd(args, latents11, latents21) + mmd(args, latents12, latents22)
        d_qz = mmd(args, z11, latents11) + mmd(args, z12, latents12) + mmd(args, z21, latents21) + mmd(args, z22, latents22)
        d_loss = dz + dq/2 - 2*d_qz/4
        d_loss.backward()
        
        l1_w, l1_b = W1(codes[0])
        l2_w, l2_b = W2(codes[1])
        clf_loss, acc = 0, 0
        for (h1_w, h1_b, h2_w, h2_b) in zip(l1_w, l1_b, l2_w, l2_b):
            h1 = (h1_w, h1_b)
            h2 = (h2_w, h2_b)
            loss, correct = train_nn(args, [h1, h2], data, targets)
            clf_loss += loss
            acc += correct
            loss.backward(retain_graph=True)
        G_loss = clf_loss / args.batch_size
        G_loss.backward()
        total_hyper_loss = G_loss #+ (gp.sum().cuda())#mean().cuda()
        
        optimE.step(); optimW1.step(); optimW2.step();
        optimE.zero_grad(); optimW1.zero_grad(), optimW2.zero_grad()
        total_loss = total_hyper_loss.item()
            
        if epoch % 100 == 0:
            acc /= args.batch_size
            print ('**************************************')
            print ('Acc: {}, MD Loss: {}, D loss: {}'.format(acc, total_hyper_loss, d_loss))
            print ('**************************************')
            #if epoch > 600:
            #    with torch.no_grad():
            #        test_far_data(netE, W1, W2)
                    #plot(netE, W1, W2, epoch)            
            #utils.save_hypernet_toy(args, [netE, netD, W1, W2], test_acc)
        if epoch > 700:
            return args, [netE, W1, W2]


if __name__ == '__main__':
    args = load_args()
    train(args)
