import torch
import argparse
import os
import random
import json
from PCA_NAB_train import train as NAB_train
from PCA_NAB_train import validate as NAB_validate
from dynamic_CUB_train import train as CUB_train
from dynamic_CUB_train import validate as CUB_validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CUB2011', type=str) # NABird CUB2011
    parser.add_argument('--gpu', default='2', type=str, help='index of GPU to use')
    parser.add_argument('--splitmode', default='easy', type=str, help='the way to split train/test data: easy/hard')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--resume',  type=str, default='', help='the model to resume')
    parser.add_argument('--evl_interval',  type=int, default=100)
    parser.add_argument('--cls_lr',  default=0.00015)
    parser.add_argument('--n_u_sample',  default=1000)

    opt = parser.parse_args()
    print('Running parameters:')
    print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    opt.GP_LAMBDA = 10      # Gradient penalty lambda
    opt.CENT_LAMBDA  = 1
    opt.REG_W_LAMBDA = 0.001
    opt.REG_Wz_LAMBDA = 0.0001
    opt.n_s_sample = 200

    opt.cls_epoch = 70
    opt.cls_batch_size = 50
    opt.lr = 0.0001
    opt.batchsize = 1000

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    if opt.dataset == 'NABird':
        NAB_validate(opt) if opt.resume else NAB_train(opt)
    elif opt.dataset == 'CUB2011':
        CUB_validate(opt) if opt.resume else CUB_train(opt)


if __name__ == '__main__':
    main()