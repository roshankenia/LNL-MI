import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import time
import argparse
from data.cifar import CIFAR10, CIFAR100
from train import train
from model import CNN

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# store starting time
begin = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type=str,
                    help='dir to save result txt files', default='results/')
parser.add_argument('--noise_rate', type=float,
                    help='corruption rate, should be less than 1', default=0.5)
parser.add_argument('--forget_rate', type=float,
                    help='forget rate', default=None)
parser.add_argument('--noise_type', type=str,
                    help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type=str,
                    help='mnist, cifar10, or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr


# obtain data
input_channel = 3
num_classes = 10
args.top_bn = False
args.epoch_decay_start = 80
args.n_epoch = 200
train_dataset = CIFAR10(root='./data/',
                        download=True,
                        train=True,
                        transform=transforms.ToTensor(),
                        noise_type=args.noise_type,
                        noise_rate=args.noise_rate
                        )

test_dataset = CIFAR10(root='./data/',
                            download=True,
                            train=False,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                       )

# Data Loader (Input Pipeline)
print('loading dataset...')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=args.num_workers,
                                           drop_last=True,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=args.num_workers,
                                          drop_last=True,
                                          shuffle=False)

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / \
        (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1


# Define models
print('building model...')
# create our full model
# fullModel = torchvision.models.resnet34(pretrained=False, num_classes=10)
# fullModel.cuda()

fullModel = CNN(input_channel=3, n_outputs=10)
fullModel.cuda()
fullOptimizer = torch.optim.Adam(fullModel.parameters(), lr=learning_rate)

ensembleModels = []
ensembleOptimizers = []
for k in range(8):
    # create ensemble model
    ensembleModel = torchvision.models.resnet34(
        pretrained=False, num_classes=10)
    ensembleModel.cuda()
    # create optimizer
    ensembleOptimizer = torch.optim.Adam(
        ensembleModel.parameters(), lr=learning_rate)
    # add to arrays
    ensembleModels.append(ensembleModel)
    ensembleOptimizers.append(ensembleOptimizer)


# training
for epoch in range(1, args.n_epoch):
    # adjust learning rate
    adjust_learning_rate(fullOptimizer, epoch)
    # train models
    train(train_loader, epoch, fullModel, fullOptimizer, ensembleModels,
          ensembleOptimizers, args.n_epoch, len(train_dataset), batch_size)


# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
print(f"Total runtime of the program is {timeTaken}")
