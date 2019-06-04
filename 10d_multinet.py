import matplotlib
matplotlib.use('agg')
import os
import sys
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.manifold import TSNE, MDS

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import ops
import utils
import arch.toy_expr as models


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=32, type=int, help='latent space width')
    parser.add_argument('--s', default=64, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=200001, type=int)
    parser.add_argument('--dataset', default='gaussian', type=str)
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--act', default='cos', type=str)
    parser.add_argument('--nd', default=10, type=int)
    parser.add_argument('--npts', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
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
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(10, 100)
        self.linear3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.elu(self.linear1(x))
        return self.linear3(x)


""" functional version of the actual target network """
def eval_nn_f(args, data, layers):
    h1_w, h1_b, h2_w, h2_b = layers
    x = F.linear(data, h1_w, bias=h1_b)
    if args.act == 'cos':
        x = torch.exp(-x*.5) * (2*np.pi*torch.cos(x))
    if args.act == 'relu':
        x = F.elu(x)
    x = F.linear(x, h2_w, bias=h2_b)
    return x


""" 
trains hypergan target network,
needs to match above network architectures
"""
def train_nn(args, Z, data, target, p=False):
    """ calc classifier loss on target architecture """
    data, target = data.cuda(), target.cuda()
    target = target.view(-1)
    x = eval_nn_f(args, data, Z)
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
def plot_data(args, x, y, title):
    plt.close('all')
    data_arr = [[], [], [], []]
    for (data, target) in zip(x, y):
        data_arr[target].append(data.numpy())
    data = np.stack(data_arr).reshape(-1, args.nd)
    if len(data) < 500:
        #proj = TSNE(2, perplexity=5, learning_rate=50, init='pca', random_state=0)
        proj = MDS(2, n_init=3)
    else:
        #proj = TSNE(2, perplexity=100, learning_rate=50, init='pca', random_state=0)
        proj = MDS(2, n_init=3)
    data = proj.fit_transform(data)
    i = len(data)//4
    c1 = data[:i] 
    c2 = data[i:i*2] 
    c3 = data[i*2:i*3] 
    c4 = data[i*3:] 
    plt.scatter(*zip(*c1), alpha=.5, linewidth=.1, edgecolor='k', label='c1')
    plt.scatter(*zip(*c2), alpha=.5, linewidth=.1, edgecolor='k', label='c2')
    plt.scatter(*zip(*c3), alpha=.5, linewidth=.1, edgecolor='k', label='c3')
    plt.scatter(*zip(*c4), alpha=.5, linewidth=.1, edgecolor='k', label='c4')
    plt.savefig('{}/{}'.format(args.save_dir, title))

def plot_multidata(args, x1, x2, y1, y2, title):

    data_arr1 = [[], [], [], []]
    for (data, target) in zip(x1, y1):
        data_arr1[target].append(data.numpy())
    data1 = np.stack(data_arr1).reshape(-1, args.nd)
    data_arr2 = [[], [], [], []]
    for (data, target) in zip(x2, y2):
        data_arr2[target].append(data.numpy())
    data2 = np.stack(data_arr2).reshape(-1, args.nd)
    #proj = TSNE(2, perplexity=100, learning_rate=50, init='pca', random_state=0)
    proj = MDS(2, n_init=3, random_state=123)
    data1 = proj.fit_transform(data1)
    data2 = proj.fit_transform(data2)

    i = len(data1)//4
    c1_1 = data1[:i] 
    c2_1 = data1[i:i*2] 
    c3_1 = data1[i*2:i*3] 
    c4_1 = data1[i*3:] 
    plt.scatter(*zip(*c1_1), alpha=.5, linewidth=.1, edgecolor='k', label='c1')
    plt.scatter(*zip(*c2_1), alpha=.5, linewidth=.1, edgecolor='k', label='c2')
    plt.scatter(*zip(*c3_1), alpha=.5, linewidth=.1, edgecolor='k', label='c3')
    plt.scatter(*zip(*c4_1), alpha=.5, linewidth=.1, edgecolor='k', label='c4')
    
    i = len(data2)//4
    c1_2 = data2[:i] 
    c2_2 = data2[i:i*2] 
    c3_2 = data2[i*2:i*3] 
    c4_2 = data2[i*3:] 
    plt.scatter(*zip(*c1_2), alpha=.5, linewidth=.1, edgecolor='k', label='c1')
    plt.scatter(*zip(*c2_2), alpha=.5, linewidth=.1, edgecolor='k', label='c2')
    plt.scatter(*zip(*c3_2), alpha=.5, linewidth=.1, edgecolor='k', label='c3')
    plt.scatter(*zip(*c4_2), alpha=.5, linewidth=.1, edgecolor='k', label='c4')
    
    plt.legend(loc='best')
    plt.savefig('{}/{}'.format(args.save_dir, title))



""" 
this will plot data from the 4 clusters
currently supports passing class data (x, y) and entropy - alpha
saves to some predefined folder
"""
def plot_data_entropy(args, outliers, y, reals, y_nets, ents, title):
    plt.close('all')
    x, y_out = outliers
    ents = np.array(ents)
    ents = 1 - ((ents - ents.min()) / (ents.max() - ents.min()))
    ents[ents>.7] = 1.0
    ents[ents<=0.35] = 0.
    # plot individual nets
    fig, ax = plt.subplots(4, 5, figsize=(15, 15))
    plt.suptitle('HyperGAN 10D Projection')
    model = 0
    data_r, target_r = reals
    data_r = data_r.cpu().numpy()
    #proj_out = TSNE(2, perplexity=250, learning_rate=50, init='pca', random_state=0)
    proj_out = MDS(2, n_init=3, random_state=123)
    #proj_in = TSNE(2, perplexity=args.npts//2, learning_rate=50, init='pca', random_state=0)
    proj_in = MDS(2, n_init=3, random_state=123)
    x = proj_out.fit_transform(x.cpu().numpy())
    data_r = proj_in.fit_transform(data_r)

    print ('plotting nets')
    # light color is predictions, dark is real data
    for xsub in range(3):
        for ysub in range(5):
            y_net = y_nets[model, :]
            ax[xsub, ysub].set_title('Net {}'.format(model))
            for (data, target) in zip(x, y_net):
                ax[xsub, ysub].scatter(*data, c=to_color_lt(target))
            for (data, target) in zip(data_r, target_r):
                ax[xsub, ysub].scatter(*data, c=to_color_dk(target))
            model += 1
    # plot hypernet
    ax[3, 0].set_title('HyperGAN')
    for (data, target, ent) in zip(x, y, ents):
        ax[3, 0].scatter(*data, c=to_color_lt(target), alpha=ent)
    for (data, target) in zip(data_r, target_r):
        ax[3, 0].scatter(*data, c=to_color_dk(target))

    ax[3, 1].set_title('Data')
    for (data, target) in zip(x, y_out):
        ax[3, 1].scatter(*data, c=to_color_lt(target))
    for (data, target) in zip(data_r, target_r):
        ax[3, 1].scatter(*data, c=to_color_dk(target))
    print ('saving to ', args.save_dir)
    plt.savefig(args.save_dir+'/{}'.format(title))

"""
aggregates predicted classes for plotting 
can be used for standard NN or for hypergan
implements hypergan target network as a functional 
passes whatever data to plotting
"""
def get_points(args, outliers, reals, nets, iter):
    outliers, y_out = outliers
    mixer, W1, W2 = nets
    preds, ents = [], []
    y_nets = torch.zeros(15, len(outliers))
    print ('calculating network responses for {}^2 points'.format(args.npts))
    for sample_idx, sample in enumerate(outliers):
        s = torch.randn(args.batch_size, args.s).cuda()
        codes = mixer(s)
        logits = []
        l1_w, l1_b = W1(codes[0], training=False)
        l2_w, l2_b = W2(codes[1], training=False)
        clf_loss, acc = 0, 0
        layers_all = [l1_w, l1_b, l2_w, l2_b]
        for i, layers in enumerate(zip(*layers_all)):
            data = sample.cuda()
            x = eval_nn_f(args, data, layers)
            logits.append(x)
            if i < 15: # only plot first 15 nets
                y_nets[i, sample_idx] = x.max(0)[1].item()
        y = torch.stack(logits).mean(0).view(1, 4)
        preds.append(F.softmax(y, dim=1).max(1, keepdim=True)[1].item())
        ents.append(entropy(F.softmax(torch.stack(logits), dim=1).mean(0).cpu().numpy().T))
        #probs.append(F.softmax(torch.stack(preds),dim=1).mean(0).max().item())
    plot_data_entropy(args, (outliers, y_out), preds, reals, y_nets, ents, 'gaussian_{}'.format(iter))
    

""" permutes a data and label tensor with the same permutation matrix """
def perm_data(x, y):
    perm = torch.randperm(len(x))
    x_perm = x[perm, :]
    y_perm = y[perm]
    return x_perm.cuda(), y_perm.cuda()


def get_means():
    means = []
    means.append(torch.tensor([0, 2, 1, 2, 2, 0, 1, 0, 1, 0]).float())
    means.append(torch.tensor([5, 3, 4, 4, 3, 5, 3, 4, 4, 3]).float())
    means.append(torch.tensor([6, 7, 7, 6, 7, 6, 7, 6, 7, 7]).float())
    means.append(torch.tensor([8, 9, 8, 9, 8, 9, 8, 9, 8, 9]).float()) 
    return means


def create_data(args):
    n = args.npts
    means = get_means()
    inliers, outliers = [], []
    for i in range(4):
        inliers.append(MultivariateNormal(means[i], torch.eye(10)*.005))
    for i in range(4):
        # sample from larger distribution. soap bubbles are helpful here
        outliers.append(MultivariateNormal(means[i], torch.eye(10)*2))
    inlier_samples, outlier_samples = [], []
    for i in range(4):
        inlier_samples.append(inliers[i].sample((n,)))
    for i in range(4):
        outlier_samples.append(outliers[i].sample((500,)))

    x_in = torch.stack(inlier_samples).view(-1, 10).cuda()
    x_out = torch.stack(outlier_samples).view(-1, 10).cuda()
    y_in = torch.ones(args.npts)
    y_out = torch.ones(500)
    y_in = torch.stack([y_in*0, y_in, y_in*2, y_in*3]).long().view(-1).cuda()
    y_out = torch.stack([y_out*0, y_out, y_out*2, y_out*3]).long().view(-1).cuda()
    plot_multidata(args, x_in.cpu(), x_out.cpu(), y_in.cpu(), y_out.cpu(), 'gaussian_all')
    return (x_in, y_in), (x_out, y_out)


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
    
    mixer = models.Mixer(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    print (mixer, W1, W2)

    optimE = optim.Adam(mixer.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-3)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-3)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-3)
    
    best_test_acc, best_clf_acc, best_test_loss, = 0., 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc
    args.best_clf_loss, args.best_clf_acc = np.inf, 0

    print ('==> Creating 4 Gaussians')
    real, outliers = create_data(args)
    data, targets = real
    one = torch.tensor(1.).cuda()
    mone = one * -1
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.pretrain_e is True:
        for j in range(100):
            s = torch.randn(e_batch_size, args.s).cuda()
            full_z = torch.randn(e_batch_size, args.z*2).cuda()
            codes = torch.stack(mixer(s)).view(-1, args.z*2)
            mean_loss, cov_loss = ops.pretrain_loss(codes, full_z)
            loss = mean_loss + cov_loss
            loss.backward()
            optimE.step()
            mixer.zero_grad()
            print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
                j, mean_loss.item(), cov_loss.item()))
            final = loss.item()
            if loss.item() < 0.1:
                print ('Finished Pretraining Mixer')
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
        #test_far_data(mixer, W1, W2)
        #plot(mixer, W1, W2, 0, net)
        #sys.exit(0)
    sample_fn = lambda x: torch.randn(args.batch_size, x).cuda()
    print ('==> Begin Training')
    for epoch in range(args.epochs):
        data, targets = perm_data(data, targets)
        # data, targets = create_data(args)
        s = torch.randn(args.batch_size, args.s).cuda()
        z = torch.randn(args.batch_size, args.z).cuda()
        full_z = torch.randn(args.batch_size, args.z*2).cuda()
        codes = mixer(s)
        
        z11 = torch.randn(args.batch_size, args.z).cuda()
        z12 = torch.randn(args.batch_size, args.z).cuda()
        z21 = torch.randn(args.batch_size, args.z).cuda()
        z22 = torch.randn(args.batch_size, args.z).cuda()
        latents11, latents12 = mixer(sample_fn(args.s))
        latents21, latents22 = mixer(sample_fn(args.s))
        dz = mmd(z11, z21)
        dq = mmd(latents11, latents21) + mmd(latents12, latents22)
        d_qz = mmd(z11, latents11) + mmd(z12, latents12) + mmd(z21, latents21) + mmd(z22, latents22)
        d_loss = dz + dq/2 - 2*d_qz/4
        d_loss.backward()
        
        l1_w, l1_b = W1(codes[0])
        l2_w, l2_b = W2(codes[1])
        clf_loss, acc = 0, np.zeros((args.batch_size))
        layers_all = [l1_w, l1_b, l2_w, l2_b]
        for i, layers in enumerate(zip(*layers_all)):
            loss, correct = train_nn(args, layers, data, targets)
            clf_loss += loss
            acc[i] = correct
            # loss.backward(retain_graph=True)
        G_loss = clf_loss / args.batch_size
        G_loss.backward()
        
        optimE.step()
        optimW1.step()
        optimW2.step()
        optimE.zero_grad()
        optimW1.zero_grad()
        optimW2.zero_grad()
        G_loss = G_loss.item()
            
        if epoch % 100 == 0:
            m = np.around(acc.mean(), decimals=3)
            s = np.around(acc.std(), decimals=3)
            print ('**************************************')
            print ('Acc: {}-{}, MD Loss: {}, D loss: {}'.format(m, s, G_loss, d_loss))
            print ('**************************************')
            with torch.no_grad():
                #test_far_data(mixer, W1, W2)
                get_points(args, outliers, (data, targets), [mixer, W1, W2], epoch)            
            #utils.save_hypernet_toy(args, [mixer, netD, W1, W2], test_acc)


if __name__ == '__main__':
    args = load_args()
    if args.gpu is not 0:
        torch.cuda.set_device(args.gpu)
        print ('using GPU ', torch.cuda.current_device())
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train(args)
