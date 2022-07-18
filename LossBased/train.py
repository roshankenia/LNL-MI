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

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


def train(train_loader, epoch, fullModel, fullOptimizer, ensembleModels, ensembleOptimizers, epochs, train_len, batch_size):
    train_total = 0
    train_correct = 0
    ensembleTotals = []
    ensembleCorrects = []

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
        # calculate full loss
        loss_1 = torch.average(
            torch.sum(F.cross_entropy(logits1, labels, reduce=False)))

        # do train for each ensemble model
        ensembleLosses = []
        ensemblePrec = []
        for k in range(len(ensembleModels)):
            ensembleModel = ensembleModels[k]
            logits = ensembleModel(images)
            prec, _ = accuracy(logits, labels, topk=(1, 5))
            ensembleTotals[k] += 1
            ensembleCorrects[k] += prec
            ensemblePrec.append(prec)
            # calculate loss for ensemble model
            loss = torch.average(
                torch.sum(F.cross_entropy(logits, labels, reduce=False)))
            ensembleLosses.append(loss)

        fullOptimizer.zero_grad()
        loss_1.backward()
        fullOptimizer.step()

        # now do step for ensemble models
        for k in range(len(ensembleOptimizers)):
            ensembleOptimizers[k].zero_grad()
            ensembleLosses[k].backward()
            ensembleOptimizers[k].step()

        if (i+1) % 50 == 0:
            print('Epoch [%d/%d], Iter [%d/%d]'
                  % (epoch+1, epochs, i+1, train_len//batch_size))
            print(f'Full model Accuracy:{prec1}, loss:{loss_1.data.item()}')
            for k in range(len(ensemblePrec)):
                print(
                    f'Model {k} Accuracy:{ensemblePrec[k]}, loss: {ensembleLosses[k].data.item()}')

    train_acc1 = float(train_correct)/float(train_total)
    return train_acc1


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
