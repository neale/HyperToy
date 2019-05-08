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

import ops
import utils


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=10, type=int, help='latent space width')
    parser.add_argument('--ze', default=64, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--dataset', default='gaussian', type=str)
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--nd', default=2, type=str)
    parser.add_argument('--beta', default=10, type=int)
    parser.add_argument('--l', default=10, type=int)
    parser.add_argument('--pretrain_e', default=True, type=bool)
    parser.add_argument('--exp', default='0', type=str)

    args = parser.parse_args()
    return args

def to_color_dk(y):
    if y == 0: return 'darkcyan'
    if y == 1: return 'darkmagenta'
    if y == 2: return 'darkgreen'
    if y == 3: return 'darkorange'
def to_color_lt(y):
    if y == 0: return 'cyan'
    if y == 1: return 'magenta'
    if y == 2: return 'green'
    if y == 3: return 'orange'

""" sanity check standard NN """
class Encoder(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 1)
    def forward(self, x):
        x = F.elu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(1, 100)
        self.linear2 = nn.Linear(100, 2)
    def forward(self, x):
        x = F.elu(self.linear1(x))
        return self.linear2(x)
class AE(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


""" functional version of the actual target network """
def eval_nn_f(data, layers):
    e1_w, e1_b, e2_w, e2_b, d1_w, d1_b, d2_w, d2_b = layers
    x = F.elu(F.linear(data, e1_w, bias=e1_b))
    x = F.linear(x, e2_w, bias=e2_b)
    x = F.elu(F.linear(x, d1_w, bias=d1_b))
    x = F.linear(x, d2_w, bias=d2_b)
    return x


""" 
trains hypergan target network,
needs to match above network architectures
"""
def train_nn(args, Z, data, target):
    """ calc classifier loss on target architecture """
    data, target = data.cuda(), target.cuda()
    target = target.view(-1)
    x = eval_nn_f(data, Z)

    loss = F.mse_loss(x, data)
    return loss


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
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    #plt.legend(loc='best')
    plt.savefig('{}/{}'.format(args.save_dir, title))


""" 
this will plot data from the 4 clusters
currently supports passing class data (x, y) and entropy - alpha
saves to some predefined folder
"""
def plot_data_entropy(x, y, real, preds_all, title):
    plt.close('all')
    fig, ax = plt.subplots(4, 5, figsize=(15,15))
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.suptitle('HyperGAN Autoencoding')
    model = 0
    # plot 15 subplots, one for each network prediction
    for xsub in range(3):
        for ysub in range(5):
            preds = preds_all[model, :, :]
            ax[xsub, ysub].set_title('AE {}'.format(model))
            ax[xsub, ysub].set_ylim(-5, 15)
            ax[xsub, ysub].set_xlim(-5, 15)
            for (data, target) in zip(preds, y):
                data = data.cpu().numpy()
                ax[xsub, ysub].scatter(*data, c=to_color_lt(target))
            model += 1
            data_r, target_r = real
            for (data, target) in zip(data_r, target_r):
                data = data.cpu().numpy()
                ax[xsub, ysub].scatter(*data, c=to_color_dk(target), alpha=0.1)

    ax[3, 0].set_title('Average AE'.format(model))
    ax[3, 0].set_ylim(-5, 15)
    ax[3, 0].set_xlim(-5, 15)       
    for (data, target) in zip(x, y):
        data = data.cpu().numpy()
        ax[3, 0].scatter(*data, c=to_color_lt(target))#, alpha=ent)
    x, y = real
    for (data, target) in zip(x, y):
        data = data.cpu().numpy()
        ax[3, 0].scatter(*data, c=to_color_dk(target), alpha=0.1)

    print ('saving to ', args.save_dir)
    plt.savefig(args.save_dir+'/{}'.format(title))

"""
aggregates predicted classes for plotting 
can be used for standard NN or for hypergan
implements hypergan target network as a functional 
passes whatever data to plotting
"""
def get_points(mixer, hyperAE, iter, ae=None):
    E1, E2, D1, D2 = hyperAE
    points, targets, ents, probs = [], [], [], []
    data, t = create_data(n=50)
    z = torch.randn(args.batch_size, args.ze).cuda()
    codes = mixer(z)
    l1w, l1b = E1(codes[0], training=False)
    l2w, l2b = E2(codes[1], training=False)
    l3w, l3b = D1(codes[2], training=False)
    l4w, l4b = D2(codes[3], training=False)
    layers_all = [l1w, l1b, l2w, l2b, l3w, l3b, l4w, l4b]
    preds_all = torch.zeros((15, 150, 2))
    for i, p in enumerate(data):
        preds = []
        for (layers) in zip(*layers_all):
            if ae is not None:
                x = ae(p)
            else:
                x = eval_nn_f(p, layers)
            preds.append(x)
            preds_all[:, i, :] = x
        points.append(p)
        #ents.append(entropy(F.softmax(torch.stack(preds), dim=1).mean(0).cpu().numpy().T))
    x = preds_all.mean(0)
    plot_data_entropy(x, t, (data, t), preds_all, 'gaussian_{}'.format(iter))
    

""" permutes a data and label tensor with the same permutation matrix """
def perm_data(x, y):
    perm = torch.randperm(len(x))
    x_perm = x[perm, :]
    y_perm = y[perm]
    return x_perm.cuda(), y_perm.cuda()


def create_data(n=2):
    dist1 = MultivariateNormal(torch.tensor([4.0, 4.0]), torch.eye(2)*.05)
    dist2 = MultivariateNormal(torch.tensor([6.0, 4.0]), torch.eye(2)*.05)
    dist3 = MultivariateNormal(torch.tensor([5.0, 7.0]), torch.eye(2)*.05)
    p1 = dist1.sample((n,))
    p2 = dist2.sample((n,))
    p3 = dist3.sample((n,))
    x = torch.stack([p1, p2, p3]).view(-1, 2).cuda()
    y_base = torch.ones(n)
    y = torch.stack([y_base*0, y_base, y_base*2]).long().view(-1).cuda()
    plot_data(x.cpu(), y.cpu(), 'gaussian_2')
    return x, y


def train(args):
    
    mixer = models.MixerS(args).cuda()
    E1 = models.GeneratorE1(args).cuda()
    E2 = models.GeneratorE2(args).cuda()
    D1 = models.GeneratorD1(args).cuda()
    D2 = models.GeneratorD2(args).cuda()
    netD = models.DiscriminatorZ(args).cuda()
    print (mixer, E1, E2, D1, D2)

    optimE = optim.Adam(mixer.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
    optimE1 = optim.Adam(E1.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
    optimE2 = optim.Adam(E2.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
    optimD1 = optim.Adam(D1.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
    optimD2 = optim.Adam(D2.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
    optimD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    
    best_test_acc, best_clf_acc, best_test_loss, = 0., 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc
    args.best_clf_loss, args.best_clf_acc = np.inf, 0

    print ('==> Creating 4 Gaussians')
    data, targets = create_data()
    one = torch.tensor(1.).cuda()
    mone = one * -1
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.pretrain_e is True:
        for j in range(100):
            x = torch.randn(e_batch_size, args.ze).cuda()
            qz = torch.randn(e_batch_size, args.z*4).cuda()
            codes = torch.stack(mixer(x)).view(-1, args.z*4)
            mean_loss, cov_loss = ops.pretrain_loss(codes, qz)
            loss = mean_loss + cov_loss
            loss.backward()
            optimE.step()
            mixer.zero_grad()
            print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
                j, mean_loss.item(), cov_loss.item()))
            final = loss.item()
            if loss.item() < 0.1:
                print ('Finished Pretraining Encoder')
                break

    print ('==> Begin Training')
    for epoch in range(args.epochs):
        data, targets = perm_data(data, targets)
        z = torch.randn(args.batch_size, args.ze).cuda()
        ze = torch.randn(args.batch_size, args.z).cuda()
        qz = torch.randn(args.batch_size, args.z*4).cuda()
        optimE.zero_grad()
        optimD.zero_grad()
        
        codes = mixer(z)
        noise = torch.randn(args.batch_size, args.ze*4)
        log_pz = ops.log_density(ze, 2).view(-1, 1)
        d_loss, d_q = ops.calc_d_loss(args, netD, ze, codes, log_pz)
        d_loss.backward(retain_graph=True)
        optimD.step()
        optimE.step()

        
        optimE.zero_grad()
        optimD.zero_grad()
        optimE1.zero_grad()
        optimE2.zero_grad()
        optimD1.zero_grad()
        optimD2.zero_grad()

        l1w, l1b = E1(codes[0])
        l2w, l2b = E2(codes[1])
        l3w, l3b = D1(codes[2])
        l4w, l4b = D2(codes[3])
        layers_all = [l1w, l1b, l2w, l2b, l3w, l3b, l4w, l4b]
        clf_loss = 0
        for i, (layers) in enumerate(zip(*layers_all)):
            loss = train_nn(args, layers, data, targets)
            clf_loss += loss
        G_loss = clf_loss / args.batch_size
        G_loss.backward()
        total_hyper_loss = G_loss #+ (gp.sum().cuda())#mean().cuda()
        
        optimE.step()
        optimE1.step() 
        optimE2.step()
        optimD1.step()
        optimD2.step()

        total_loss = total_hyper_loss.item()
            
        if epoch % 2 == 0:
            print ('**************************************')
            print ('AE-MD Loss: {}, D loss: {}'.format(total_hyper_loss, d_loss))
            print ('**************************************')
            #if epoch > 100:
            with torch.no_grad():
                #test_far_data(mixer, W1, W2)
                get_points(mixer, [E1, E2, D1, D2], epoch)            
            #utils.save_hypernet_toy(args, [mixer, netD, W1, W2], test_acc)


if __name__ == '__main__':
    args = load_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #import arch.toy_tiny_gen as models
    import arch.ae_models as models
    train(args)
