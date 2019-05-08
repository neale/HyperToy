import matplotlib
matplotlib.use('agg')
import os
import sys
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import args as arg
import netdef, datagen
import ops, mnist, test, utils
import stats_mnist as stats
import arch.mnist_clf as mnist_clf
import ensemble

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=32, type=int, help='latent space width')
    parser.add_argument('--ze', default=32, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--dataset', default='gaussian', type=str)
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--beta', default=10, type=int)
    parser.add_argument('--l', default=10, type=int)
    parser.add_argument('--pretrain_e', default=True, type=bool)
    parser.add_argument('--exp', default='0', type=str)

    args = parser.parse_args()
    return args


""" sanity check standard NN """
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(2, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))
        return self.linear5(x)


""" functional version of the actual target network """
def eval_nn_f(data, layers):
    h1, h2, h3, h4, h5 = layers
    h1_w, h1_b = h1
    h2_w, h2_b = h2
    h3_w, h3_b = h2
    h4_w, h4_b = h2
    h5_w, h5_b = h2
    x = F.elu(F.linear(data, h1_w, bias=h1_b))
    x = F.elu(F.linear(data, h2_w, bias=h2_b))
    x = F.elu(F.linear(data, h3_w, bias=h3_b))
    x = F.elu(F.linear(data, h4_w, bias=h4_b))
    x = F.linear(x, h5_w, bias=h5_b)
    return x


""" 
trains hypergan target network,
needs to match above network architectures
"""
def train_nn(args, Z, data, target, p=False):
    """ calc classifier loss on target architecture """
    data, target = data.cuda(), target.cuda()
    target = target.view(-1)
    x = eval_nn_f(data, Z)
    loss = F.cross_entropy(x, target)
    pred = x.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    if p:
        print ('target', target, 'pred', pred.view(-1))
       
    return loss, correct


""" 
barebones plotting function
plots class labels and thats it
"""
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
    plt.savefig('{}/{}'.format(args.save_dir, title))


""" 
this will plot data from the 4 clusters
currently supports passing class data (x, y) and entropy - alpha
saves to some predefined folder
"""
def plot_data_entropy(x, y, ents, title):
    plt.close('all')
    ents = np.array(ents)
    #prob = np.array(prob)
    ents = 1 - ((ents - ents.min()) / (ents.max() - ents.min()))
    ents[ents>.7] = 1.0
    ents[ents<=0.35] = 0.

    def to_color(y):
        if y == 0: return 'b'
        if y == 1: return 'y'
        if y == 2: return 'g'
        if y == 3: return 'r'

    for (data, target, ent) in zip(x, y, ents):
        plt.scatter(*data, c=to_color(target), alpha=ent)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title('HyperGAN')
    print ('saving to ', args.save_dir)
    plt.savefig(args.save_dir+'/{}'.format(title))

"""
aggregates predicted classes for plotting 
can be used for standard NN or for hypergan
implements hypergan target network as a functional 
passes whatever data to plotting
"""
def get_points(netE, W1, W2, W3, W4, W5, iter, net=None):
    points, targets, ents, probs = [], [], [], []
    for x1 in np.linspace(-15, 25, 100):
        for x2 in np.linspace(-15, 25, 100):
            z = torch.randn(args.batch_size, args.ze).cuda()
            codes = netE(z)
            preds = []
            l1_w, l1_b = W1(codes[0], training=False)
            l2_w, l2_b = W2(codes[1], training=False)
            l3_w, l3_b = W3(codes[2], training=False)
            l4_w, l4_b = W4(codes[3], training=False)
            l5_w, l5_b = W5(codes[4], training=False)
            clf_loss, acc = 0, 0
            for (h1_w, h1_b, h2_w, h2_b, h3_w, h3_b, h4_w, h4_b, h5_w, h5_b) in zip(l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, l4_w, l4_b, l5_w, l5_b):
                data = torch.tensor([x1, x2]).cuda()
                if net is not None:
                    x = net(data)
                else:
                    x = eval_nn_f(data, [(h1_w, h1_b), (h2_w, h2_b), (h3_w, h3_b), (h4_w, h4_b), (h5_w, h5_b)])
                preds.append(x)
            points.append((x1, x2))
            y = torch.stack(preds).mean(0).view(1, 4)
            targets.append(F.softmax(y, dim=1).max(1, keepdim=True)[1].item())
            ents.append(entropy(F.softmax(torch.stack(preds), dim=1).mean(0).cpu().numpy().T))
            #probs.append(F.softmax(torch.stack(preds),dim=1).mean(0).max().item())
    plot_data_entropy(points, targets, ents, 'gaussian_{}'.format(iter))
    

""" permutes a data and label tensor with the same permutation matrix """
def perm_data(x, y):
    perm = torch.randperm(len(x))
    x_perm = x[perm, :]
    y_perm = y[perm]
    return x_perm.cuda(), y_perm.cuda()


def create_data(args):
    dist1 = MultivariateNormal(torch.tensor([4.2, 6.2]), torch.eye(2)*.09)
    dist2 = MultivariateNormal(torch.tensor([4.8, 5.2]), torch.eye(2)*0.06)
    dist3 = MultivariateNormal(torch.tensor([6., 6.0]), torch.eye(2)*.09)
    dist4 = MultivariateNormal(torch.tensor([6., 3.9]), torch.eye(2)*0.08)
    p1 = dist1.sample((25,))
    p2 = dist2.sample((25,))
    p3 = dist3.sample((25,))
    p4 = dist4.sample((25,))
    x = torch.stack([p1, p2, p3, p4]).view(-1, 2).cuda()
    y_base = torch.ones(25)
    y = torch.stack([y_base*0, y_base, y_base*2, y_base*3]).long().view(-1).cuda()
    plot_data(x.cpu(), y.cpu(), 'gaussian_2')
    return x, y


def mmd(x, y):
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


def train(args):
    
    netE = models.EncoderNoBN(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    W3 = models.GeneratorW3(args).cuda()
    W4 = models.GeneratorW4(args).cuda()
    W5 = models.GeneratorW5(args).cuda()
    print (netE, W1, W2, W3, W4, W5)

    optimE = optim.Adam(netE.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-2)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-2)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-2)
    optimW3 = optim.Adam(W3.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-2)
    optimW4 = optim.Adam(W4.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-2)
    optimW5 = optim.Adam(W5.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-2)
    
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
        dz = mmd(z11, z21)
        dq = mmd(latents11, latents21) + mmd(latents12, latents22)
        d_qz = mmd(z11, latents11) + mmd(z12, latents12) + mmd(z21, latents21) + mmd(z22, latents22)
        d_loss = dz + dq/2 - 2*d_qz/4
        d_loss.backward()
        
        l1_w, l1_b = W1(codes[0])
        l2_w, l2_b = W2(codes[1])
        l3_w, l3_b = W3(codes[2])
        l4_w, l4_b = W4(codes[3])
        l5_w, l5_b = W5(codes[4])
        clf_loss, acc = 0, 0
        for (h1_w, h1_b, h2_w, h2_b, h3_w, h3_b, h4_w, h4_b, h5_w, h5_b) in zip(l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, l4_w, l4_b, l5_w, l5_b):
            h1 = (h1_w, h1_b)
            h2 = (h2_w, h2_b)
            h3 = (h3_w, h3_b)
            h4 = (h4_w, h4_b)
            h5 = (h5_w, h5_b)
            loss, correct = train_nn(args, [h1, h2, h3, h4, h5], data, targets)
            clf_loss += loss
            acc += correct
            loss.backward(retain_graph=True)
        G_loss = clf_loss / args.batch_size
        G_loss.backward()
        total_hyper_loss = G_loss #+ (gp.sum().cuda())#mean().cuda()
        
        optimE.step(); optimW1.step(); optimW2.step(); optimW3.step(); optimW4.step(); optimW5.step()
        optimE.zero_grad();
        optimW1.zero_grad()
        optimW2.zero_grad()
        optimW3.zero_grad()
        optimW4.zero_grad()
        optimW5.zero_grad()
        total_loss = total_hyper_loss.item()
            
        if epoch % 100 == 0:
            acc /= args.batch_size
            print ('**************************************')
            print ('Acc: {}, MD Loss: {}, D loss: {}'.format(acc, total_hyper_loss, d_loss))
            print ('**************************************')
            #if epoch > 100:
            with torch.no_grad():
                #test_far_data(netE, W1, W2)
                get_points(netE, W1, W2, W3, W4, W5, epoch)            
            #utils.save_hypernet_toy(args, [netE, netD, W1, W2], test_acc)


if __name__ == '__main__':
    args = load_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    import arch.toy_expr_gen as models
    train(args)
