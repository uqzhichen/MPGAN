import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
from termcolor import cprint
from time import gmtime, strftime
import numpy as np
import argparse
import os
import random
import json
from PCA_dataset import FeatDataLayer, LoadDataset
from PCA_models import _netD, _netG, _param, LINEAR_LOGSOFTMAX, weights_init

import classifier_voting

def train(opt):
    param = _param()
    dataset = LoadDataset(opt)
    param.X_dim = dataset.feature_dim

    data_layer = FeatDataLayer(dataset.labels_train, dataset.pfc_feat_data_train, opt)

    # initialize model
    netGs = []
    netDs = []
    parts = 6 #if opt.dataset == "CUB2011" else 6

    for part in range(parts):
        netGs.append(_netG(dataset.text_dim, 512).cuda().apply(weights_init))

        netDs.append(_netD(dataset.train_cls_num, 512).cuda().apply(weights_init))

    exp_info = 'CUB_EASY' if opt.splitmode == 'easy' else 'CUB_HARD'
    exp_params = 'Eu{}_Rls{}_RWz{}'.format(opt.CENT_LAMBDA , opt.REG_W_LAMBDA, opt.REG_Wz_LAMBDA)

    out_dir  = 'out/{:s}'.format(exp_info)
    out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir  = out_subdir + '/log_{:s}.txt'.format(exp_info)
    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    part_cls_centrild = torch.from_numpy(dataset.part_cls_centrild.astype('float32')).cuda()

    # initialize optimizers
    optimizerGs = []
    optimizerDs = []
    for netG in netGs:
        optimizerGs.append(optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9)))
    for netD in netDs:
        optimizerDs.append(optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9)))

    for it in range(start_step, 30000+1):
        """ Discriminator """
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']             # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i,:] for i in labels])
            text_feat = torch.from_numpy(text_feat.astype('float32')).cuda()
            X = torch.from_numpy(feat_data).cuda()
            y_true = torch.from_numpy(labels.astype('int')).cuda()
            z = torch.randn(opt.batchsize, param.z_dim).cuda()

            for part in range(parts):
                z = torch.randn(opt.batchsize, param.z_dim).cuda()
                D_real, C_real = netDs[part](X[:, part*512:(part+1)*512])
                D_loss_real = torch.mean(D_real)
                C_loss_real = F.cross_entropy(C_real, y_true)
                DC_loss = -D_loss_real + C_loss_real
                DC_loss.backward()

                G_sample = netGs[part](z, text_feat)
                D_fake, C_fake = netDs[part](G_sample)
                D_loss_fake = torch.mean(D_fake)
                C_loss_fake = F.cross_entropy(C_fake, y_true)
                DC_loss = D_loss_fake + C_loss_fake
                DC_loss.backward()

                grad_penalty = calc_gradient_penalty(opt.batchsize, netDs[part], X.data[:, part*512:(part+1)*512],
                                                     G_sample.data, opt.GP_LAMBDA)
                grad_penalty.backward()

                Wasserstein_D = D_loss_real - D_loss_fake
                # writer.add_scalar("Wasserstein_D"+str(part), Wasserstein_D.item(), it)

                optimizerDs[part].step()
                netGs[part].zero_grad()
                netDs[part].zero_grad()


        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = torch.from_numpy(text_feat.astype('float32')).cuda()

            X = torch.from_numpy(feat_data).cuda()
            y_true = torch.from_numpy(labels.astype('int')).cuda()

            for part in range(parts):
                z = torch.randn(opt.batchsize, param.z_dim).cuda()
                G_sample = netGs[part](z, text_feat)
                # G_sample_all[:, part*512:(part+1)*512] = G_sample
                D_fake, C_fake = netDs[part](G_sample)
                _, C_real = netDs[part](X[:, part*512:(part+1)*512])

                G_loss = torch.mean(D_fake)
                C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2
                GC_loss = -G_loss + C_loss
                # writer.add_scalar("GC_loss"+str(part), GC_loss.item(), it)

                Euclidean_loss = torch.tensor([0.0]).cuda()
                if opt.REG_W_LAMBDA != 0:
                    for i in range(dataset.train_cls_num):
                        sample_idx = (y_true == i).data.nonzero().squeeze()
                        if sample_idx.numel() == 0:
                            Euclidean_loss += 0.0
                        else:
                            G_sample_cls = G_sample[sample_idx, :]
                            Euclidean_loss += (G_sample_cls.mean(dim=0) - part_cls_centrild[i][part]).pow(
                                2).sum().sqrt()
                    Euclidean_loss *= 1.0 / dataset.train_cls_num * opt.CENT_LAMBDA

                # ||W||_2 regularization
                reg_loss = torch.Tensor([0.0]).cuda()
                if opt.REG_W_LAMBDA != 0:

                    for name, p in netGs[part].named_parameters():
                        if 'weight' in name:
                            reg_loss += p.pow(2).sum()
                    reg_loss.mul_(opt.REG_W_LAMBDA)

                # writer.add_scalar("reg_Wz_loss"+str(part), reg_Wz_loss.item(), it)
                all_loss = GC_loss + Euclidean_loss + reg_loss
                all_loss.backward()
                optimizerGs[part].step()

        if it % opt.evl_interval == 0 and it >= 500:
            print(it)
            for part in range(parts):
                netGs[part].eval()
            train_classifier(opt, param, dataset, netGs)
            for part in range(parts):
                netGs[part].train()


def train_classifier(opt, param, dataset, netGs):

    # n-way classifier
    num_classes = dataset.test_cls_num

    # initialize classifiers
    clfs = []
    for i in range(6):
        clf = LINEAR_LOGSOFTMAX(512, num_classes)
        clf.apply(weights_init)
        clfs.append(clf)

    text_dim = dataset.train_text_feature.shape[1]
    # text_feat = torch.zeros([0, text_dim]).cuda()

    # prepare text features in order
    train_Y = []

    train_sample_n =  opt.n_u_sample * dataset.test_cls_num
    train_X = torch.zeros((train_sample_n, dataset.feature_dim))
    for i in range(dataset.test_cls_num):
        text_feat_tmp = torch.from_numpy(np.tile(dataset.test_text_feature[i].astype('float32'), (opt.n_u_sample, 1))).cuda()
        with torch.no_grad():
            for part in range(len(netGs)):
                z = torch.randn(opt.n_u_sample, param.z_dim).cuda()
                train_X[i*opt.n_u_sample:(i+1)*opt.n_u_sample, part * 512:(part + 1) * 512] = netGs[part](z, text_feat_tmp)
        train_Y = train_Y + [i] * opt.n_u_sample
    train_Y = torch.tensor(train_Y)

    torch.cuda.empty_cache()
    a = classifier_voting.CLASSIFIER(opt, clfs, dataset, train_X, train_Y, netGs)
    torch.cuda.empty_cache()


def validate(opt):
    param = _param()
    dataset = LoadDataset(opt)
    param.X_dim = dataset.feature_dim

    # data_layer = FeatDataLayer(dataset.labels_train, dataset.pfc_feat_data_train, opt)

    # initialize model
    netGs = []

    checkpoint = torch.load(opt.resume)

    parts = 6
    for part in range(parts):
        netG = _netG(400, 512).cuda()
        netG.load_state_dict(checkpoint['state_dict_G'+str(part)])
        netGs.append(netG)

    train_classifier(opt, param, dataset, netGs)

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(batchsize, netD, real_data, fake_data, GP_LAMBDA):
    alpha = torch.rand(batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()

