import math
import torch
import torch.nn.functional as F
import torch.autograd as autograd

def batch_zero_grad(modules):
    for module in modules:
        module.zero_grad()


def batch_update_optim(optimizers):
    for optimizer in optimizers:
        optimizer.step()


def free_params(modules):
    for module in modules:
        for p in module.parameters():
            p.requires_grad = False


def frozen_params(modules):
    for module in modules:
        for p in module.parameters():
            p.requires_grad = False


def pretrain_loss(encoded, noise):
    mean_z = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_z, mean_e)
    cov_z = torch.matmul((noise-mean_z).transpose(0, 1), noise-mean_z)
    cov_z /= 999
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 999
    cov_loss = F.mse_loss(cov_z, cov_e)
    return mean_loss, cov_loss



def log_density(z, z_var):
    z_dim = z.size(1)
    z = -(z_dim/2)*math.log(2*math.pi*z_var) + z.pow(2).sum(1).div(-2*z_var)
    return z.cuda()


def calc_gradient_penalty_layer(prior, netG, mixer):
    code = mixer(prior)
    gen_layer = netG(code)
    penalty = lambda x: ((x.norm(2, dim=1) - 1) ** 2).mean()
    ones = torch.ones(gen_layer.size()).cuda()
    gradients = autograd.grad(outputs=gen_layer,
            inputs=code,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
    p = torch.tensor(penalty(gradients))
    norms = torch.tensor(gradients.norm())
    return p, gradients, norms


def calc_d_loss(args, netD, z, codes, log_pz, cifar=False):
    dim = args.batch_size * len(codes)
    zeros = torch.zeros(dim, 1, requires_grad=True).cuda()
    ones = torch.ones(args.batch_size, 1, requires_grad=True).cuda()
    d_z = netD(z)
    codes = torch.stack(codes).view(-1, args.z)
    d_codes = netD(codes)
    log_pz_ = log_density(torch.ones(dim, 1), 2).view(-1, 1)
    d_loss = F.binary_cross_entropy_with_logits(d_z+log_pz, ones) + \
             F.binary_cross_entropy_with_logits(d_codes+log_pz_, zeros)
    total_loss = d_loss
    return total_loss, d_codes
