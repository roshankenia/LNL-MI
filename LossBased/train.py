import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import time
import argparse
from data.cifar import CIFAR10, CIFAR100
from loss import cross_entropy_loss, low_loss_over_epochs_labels, cross_entropy_loss_update, no_split

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


def train(train_loader, epoch, fullModel, fullOptimizer, epochs, train_len, batch_size, epochLabels):
    train_total = 0
    train_correct = 0
    totalRelabelCount = 0
    totalCorrectRelabelCount = 0
    totalIncorrectRelabelCount = 0
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        # if i > args.num_iter_per_epoch:
        #     break

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits1 = fullModel(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1

        purity_ratio_clean = 0
        purity_ratio_noisy = 0
        num_clean = 0
        num_noisy = 0

        # calculate loss
        fullLoss = None
        relabelCount = None
        if epoch < 10:
            fullLoss, relabelCount = cross_entropy_loss_update(
                logits1, labels, epochLabels, ind, epoch)
        else:
            fullLoss, purity_ratio_clean, purity_ratio_noisy, num_clean, num_noisy, relabelCount = low_loss_over_epochs_labels(
                logits1, labels, epochLabels, ind, epoch, i)
            # fullLoss, purity_ratio_clean, purity_ratio_noisy, num_clean, num_noisy, relabelCount = no_split(
            #     logits1, labels, epochLabels, ind)

        totalRelabelCount += relabelCount[0]
        totalCorrectRelabelCount += relabelCount[1]
        totalIncorrectRelabelCount += relabelCount[2]

        # fullLoss = loss_over_epochs(logits1, labels, epochLabels)

        # # find loss for full model
        # fullLoss = None
        # if epoch < 5:
        #     fullLoss = cross_entropy_loss(logits1, labels)
        # else:
        #     fullLoss = avg_loss(logits1, labels)

        fullOptimizer.zero_grad()
        fullLoss.backward()
        fullOptimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d]'
                  % (epoch+1, epochs, i+1, train_len//batch_size))
            print(
                f'\tFull model Accuracy:{prec1.data.item()}, loss:{fullLoss.data.item()}')
            print(f'\tNumber clean:{num_clean}, Number noisy:{num_noisy}')
            print(
                f'\tClean purity ratio:{purity_ratio_clean}, Noisy purity ratio:{purity_ratio_noisy}')
    print(
        f'Total Relabeled:{totalRelabelCount}, Correctly Relabeled:{totalCorrectRelabelCount}, Incorrectly Relabeled: {totalIncorrectRelabelCount}')
    train_acc1 = float(train_correct)/float(train_total)
    return train_acc1
# def train(train_loader, epoch, fullModel, fullOptimizer, ensembleModels, ensembleOptimizers, epochs, train_len, batch_size):
#     train_total = 0
#     train_correct = 0
#     ensembleTotals = np.zeros(len(ensembleModels))
#     ensembleCorrects = np.zeros(len(ensembleModels))

#     for i, (images, labels, indexes) in enumerate(train_loader):
#         ind = indexes.cpu().numpy().transpose()
#         # if i > args.num_iter_per_epoch:
#         #     break

#         images = Variable(images).cuda()
#         labels = Variable(labels).cuda()

#         # Forward + Backward + Optimize
#         logits1 = fullModel(images)
#         prec1, _ = accuracy(logits1, labels, topk=(1, 5))
#         train_total += 1
#         train_correct += prec1

#         # do train for each ensemble model
#         ensemblePreds = []
#         ensembleLosses = []
#         ensemblePrec = []
#         for k in range(len(ensembleModels)):
#             ensembleModel = ensembleModels[k]
#             logits = ensembleModel(images)
#             prec, _ = accuracy(logits, labels, topk=(1, 5))
#             ensembleTotals[k] += 1
#             ensembleCorrects[k] += prec
#             ensemblePrec.append(prec)
#             # calculate loss for ensemble model
#             loss = F.cross_entropy(logits, labels)/len(labels)

#             ensembleLosses.append(loss)
#             ensemblePreds.append(logits.unsqueeze(0))
#         # put all predictions into one tensor
#         ensemblePreds = torch.cat(ensemblePreds)
#         ensemblePredsCopy = ensemblePreds.clone()
#         # find loss for full model
#         fullLoss = loss_co_ensemble_teaching(
#             logits1, ensemblePredsCopy, labels)

#         fullOptimizer.zero_grad()
#         fullLoss.backward(retain_graph=True)
#         fullOptimizer.step()

#         # # now do step for ensemble models
#         for k in range(len(ensembleOptimizers)):
#             ensembleOptimizers[k].zero_grad()
#             ensembleLosses[k].backward()
#             ensembleOptimizers[k].step()

#         if (i+1) % 50 == 0:
#             print('Epoch [%d/%d], Iter [%d/%d]'
#                   % (epoch+1, epochs, i+1, train_len//batch_size))
#             print(
#                 f'\tFull model Accuracy:{prec1}, loss:{fullLoss.data.item()}')
#             for k in range(len(ensemblePrec)):
#                 print(
#                     f'\tModel {k} Accuracy:{ensemblePrec[k]}, loss: {ensembleLosses[k].data.item()}')

#     train_acc1 = float(train_correct)/float(train_total)
#     return train_acc1


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
