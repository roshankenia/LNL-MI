import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import sys
from utils.labels import LowLossLabels
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# Loss functions


def loss_coteaching_with_relabeling(y_1, y_2, t, indices, combinedLabels, cur_time):

    current_target, noise_or_not = combinedLabels.getLabelsOnly(indices)
    current_target = current_target.to(t.device)
    combined_logits = (y_1 + y_2)/2
    # calculate cross-entropy loss using combined logits
    combined_cross_entropy_loss = F.cross_entropy(
        combined_logits, current_target, reduction='none')
    # update our labels
    combinedLabels.update(
        combined_logits, combined_cross_entropy_loss.cpu(), indices, cur_time)

    # get new target
    current_target, noise_or_not = combinedLabels.getLabelsOnly(indices)
    current_target = current_target.to(t.device)

    loss_1 = F.cross_entropy(y_1, current_target, reduce=False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, current_target, reduce=False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    # find number of samples to use
    num_use = torch.nonzero(combined_cross_entropy_loss <
                            combined_cross_entropy_loss.mean()).shape[0]

    pure_ratio_1 = np.sum(
        noise_or_not[ind_1_sorted[:num_use]])/float(num_use)
    pure_ratio_2 = np.sum(
        noise_or_not[ind_2_sorted[:num_use]])/float(num_use)

    ind_1_update = ind_1_sorted[:num_use]
    ind_2_update = ind_2_sorted[:num_use]
    # exchange
    loss_1_update = F.cross_entropy(
        y_1[ind_2_update], current_target[ind_2_update])
    # print('before:', loss_1_update)
    # print('after:', torch.sum(loss_1_update)/num_remember)
    loss_2_update = F.cross_entropy(
        y_2[ind_1_update], current_target[ind_1_update])

    return torch.sum(loss_1_update)/num_use, torch.sum(loss_2_update)/num_use, pure_ratio_1, pure_ratio_2


def combined_relabel(y_1, y_2, t, indices, combinedLabels, cur_time):

    # calculate combined logits
    combined_logits = (y_1 + y_2)/2
    # for i in range(len(combined_logits)):
    #     print('1', y_1[i], y_2[i], combined_logits[i])
    #     print('2', torch.argmax(y_1[i]), torch.argmax(y_2[i]),
    #           torch.argmax(combined_logits[i]))
    # begin = time.time()
    current_target, noise_or_not = combinedLabels.getLabelsOnly(indices)
    # end = time.time()
    # print('current_target:', (end-begin))

    # calculate cross-entropy loss using combined logits
    combined_cross_entropy_loss = F.cross_entropy(
        combined_logits, current_target.to(t.device), reduction='none')

    # update our labels
    # begin = time.time()
    combinedLabels.update(
        combined_logits, combined_cross_entropy_loss.cpu(), indices, cur_time)
    # end = time.time()
    # print('combinedLabels:', (end-begin))
    # obtain clean labels
    # begin = time.time()
    useIndices_1, useLabels_1, useActualIndices_1, useIndices_2, useLabels_2, useActualIndices_2, lowLossCount, consistentCount, unusedCount, lowLossClean, consistentClean, unusedClean = combinedLabels.getLabels(
        y_1.cpu(), y_2.cpu(), combined_cross_entropy_loss.cpu(), indices)

    # useIndices_1, useLabels_1, useActualIndices_1, useIndices_2, useLabels_2, useActualIndices_2, lowLossCount, consistentCount, unusedCount, lowLossClean, consistentClean, unusedClean = combinedLabels.getBasedOnCount(
    #     indices)
    # end = time.time()
    # print('clean labels:', (end-begin))
    # use half labels to update model 1 while other half to update model 2
    loss_1 = F.cross_entropy(y_1[useIndices_1], useLabels_1.to(t.device))
    loss_2 = F.cross_entropy(y_2[useIndices_2], useLabels_2.to(t.device))

    return loss_1/len(useActualIndices_1), loss_2/len(useActualIndices_2), lowLossCount, consistentCount, unusedCount, lowLossClean, consistentClean, unusedClean


def cross_entropy_with_update(y_1, y_2, t, indices, combinedLabels, cur_time):
    # calculate combined logits
    combined_logits = (y_1 + y_2)/2

    # begin = time.time()
    current_target, noise_or_not = combinedLabels.getLabelsOnly(indices)
    # end = time.time()
    # print('current_target:', (end-begin))

    # calculate cross-entropy loss using combined logits
    combined_cross_entropy_loss = F.cross_entropy(
        combined_logits, current_target.to(t.device), reduction='none')

    # update our labels
    # begin = time.time()
    combinedLabels.update(
        combined_logits, combined_cross_entropy_loss.cpu(), indices, cur_time)
    # end = time.time()
    # print('combinedLabels:', (end-begin))

    # just use cross-entropy for our step
    loss_1 = F.cross_entropy(y_1, t)
    loss_2 = F.cross_entropy(y_2, t)

    return loss_1/len(t), loss_2/len(t)
