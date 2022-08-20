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
from loss import loss_coteaching

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


def train(train_loader, epoch, model_1, optimizer_1, model_2, optimizer_2, epochs, train_len, batch_size, noise_or_not, features):
    train_total = 0
    train_correct = 0
    pure_ratio_1_list = []
    pure_ratio_2_list = []
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        noise = noise_or_not[ind]
        # if i > args.num_iter_per_epoch:
        #     break

        images = Variable(images).cuda()
        cur_labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits_1 = model_1(images)
        logits_2 = model_2(images)
        combinedLogits = (logits_1+logits_2)/2
        prec, _ = accuracy(combinedLogits, cur_labels, topk=(1, 5))
        prec1, _ = accuracy(logits_1, cur_labels, topk=(1, 5))
        prec2, _ = accuracy(logits_2, cur_labels, topk=(1, 5))
        train_total += 1
        train_correct += prec

        # add to our feature data
        features.addData(combinedLogits, ind, epoch)

        # calculate loss
        loss_1 = None
        loss_2 = None
        pure_ratio_1 = None
        pure_ratio_2 = None

        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(
            logits_1, logits_2, cur_labels, noise)

        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

        if (i+1) % 50 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Combined Accuracy: %.4F, Training Accuracy1: %.4F, Training Accuracy2: %.4F, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f'
                  % (epoch+1, epochs, i+1, train_len//batch_size, prec.data.item(), prec1, prec2, loss_1.data.item(), loss_2.data.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))
    train_acc1 = float(train_correct)/float(train_total)

    if epoch % 2 == 0:
        print(features.features[0])
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
