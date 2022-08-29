import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.autograd import Variable
import os
import sys
import time
import argparse
from data.cifar import CIFAR10, CIFAR100
from train import train
from utils.featureMap import FeatureMap
from extraction.extraction import extract_features, make_plots
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
parser.add_argument('--lr', type=float, default=0.001)
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

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

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
        print('Learning rate:', alpha_plan[epoch])
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1


# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(
    0, forget_rate**args.exponent, args.num_gradual)


# def evaluate(test_loader, model):
#     model.eval()    # Change model to 'eval' mode.
#     correct1 = 0
#     total1 = 0
#     for images, labels, _ in test_loader:
#         images = Variable(images).cuda()
#         logits1 = model(images)
#         outputs1 = F.softmax(logits1, dim=1)
#         _, pred1 = torch.max(outputs1.data, 1)
#         total1 += labels.size(0)
#         correct1 += (pred1.cpu() == labels).sum()

#     acc1 = 100*float(correct1)/float(total1)
#     return acc1
def evaluate(test_loader, model_1, model_2):
    model_1.eval()    # Change model to 'eval' mode.
    model_2.eval()
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model_1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model_2.eval()    # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model_2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()

    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def main():
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
    # Define models
    print('building model...')
    cnn1 = torchvision.models.resnet34(pretrained=False, num_classes=10)
    cnn1.cuda()
    # print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)

    cnn2 = torchvision.models.resnet34(pretrained=False, num_classes=10)
    cnn2.cuda()
    # print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    mean_pure_ratio1 = 0
    mean_pure_ratio2 = 0

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    # evaluate models with random weights
    test_acc1, test_acc2 = evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' %
          (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))

    # training
    noise_or_not = train_dataset.noise_or_not
    true_train_labels = train_dataset.train_labels
    noisy_train_labels = train_dataset.train_noisy_labels

    # obtain extracted features
    # features = extract_features(train_dataset)
    # make_plots(features, noisy_train_labels, noise_or_not, num_classes)
    # exit()

    features1 = FeatureMap(len(train_dataset.train_labels),
                           args.n_epoch, num_classes, 'Model1', history=10)
    features2 = FeatureMap(len(train_dataset.train_labels),
                           args.n_epoch, num_classes, 'Model2', history=10)
    for epoch in range(0, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = train(
            train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, rate_schedule[epoch], noise_or_not, args.n_epoch, len(train_dataset), batch_size, features1, features2)
        # evaluate models
        test_acc1, test_acc2 = evaluate(test_loader, cnn1, cnn2)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' %
              (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))

        # add time to our feature maps
        features1.addTime()
        features2.addTime()
        # make plots every 10 epochs
        if (epoch+1) % 10 == 0:
            features1.makePlot(epoch, noisy_train_labels, noise_or_not)
            features2.makePlot(epoch, noisy_train_labels, noise_or_not)


if __name__ == '__main__':
    main()

# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
print(f"Total runtime of the program is {timeTaken}")
