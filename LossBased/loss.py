import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import sys
from utils.labels import LowLossLabels

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# Loss functions


def low_loss_over_epochs_labels(y_1, t, lowest_loss, indices, noise_or_not):
    # calculate loss for full
    fullLoss = F.cross_entropy(y_1, t, reduction='none')

    # update our lowest losses
    relabelCount, noise_or_not = lowest_loss.update(
        indices, fullLoss.data.cpu(), y_1.data.cpu(), noise_or_not)

    # find indexes to sort loss
    sort_index_loss = torch.argsort(fullLoss.data)

    # find number of samples to use
    num_use = torch.nonzero(fullLoss < 0.5*fullLoss.mean()).shape[0]

    # use indexes underneath this threshold and the rest are noisy
    clean_index = sort_index_loss[:num_use]
    noisy_index = sort_index_loss[num_use:]

    # obtain clean logits and labels
    clean_logits = y_1[clean_index]
    clean_labels = t[clean_index]

    # obtain noisy logits and labels
    noisy_logits = y_1[noisy_index]
    noisy_indices = indices[noisy_index.data.cpu()]
    noisy_labels = lowest_loss.labels[noisy_indices].to(clean_labels.device)

    # make array if 0-d
    if noisy_labels.dim() == 0:
        noisy_labels = noisy_labels.unsqueeze(dim=0)

    logits_final = torch.cat((clean_logits, noisy_logits), dim=0)
    labels_final = torch.cat((clean_labels, noisy_labels), dim=0)

    # total loss calculation
    totalLoss = F.cross_entropy(logits_final, labels_final)

    # calculate how much noise in each
    purity_ratio_clean = np.sum(
        noise_or_not[indices[clean_index.data.cpu()]])/float(len(clean_labels))
    purity_ratio_noisy = np.sum(
        noise_or_not[indices[noisy_index.data.cpu()]])/float(len(noisy_labels))

    return totalLoss/len(t), purity_ratio_clean, purity_ratio_noisy, len(clean_labels), len(noisy_labels), relabelCount, noise_or_not


def loss_over_epochs(y_1, t, epochLabels):
    # update our lowest losses
    preds = epochLabels.update(y_1.data.cpu()).cuda()

    # calculate loss using low loss predictions
    totalLoss = F.cross_entropy(preds, t)

    return totalLoss/len(t)


def loss_co_ensemble_teaching(y_1, ensemble_y, t):
    # calculate loss for full
    fullLoss = F.cross_entropy(y_1, t)

    # first find average y_pred for ensemble
    y_ensemble_avg = torch.mean(ensemble_y, 0)
    # calculate loss for ensemble
    ensembleLoss = F.cross_entropy(y_ensemble_avg, t)

    totalLoss = 0.5 * fullLoss + 0.5 * ensembleLoss

    return totalLoss/len(t)


def avg_loss(y_1, t):
    # calculate loss on all samples
    loss = F.cross_entropy(y_1, t, reduction='none')

    # find indexes to sort loss
    sort_index_loss = torch.argsort(loss.data)

    # find number of samples to use
    num_use = torch.nonzero(loss < loss.mean()).shape[0]

    # print(f'Using {num_use} out of {len(t)}')

    # use indexes underneath this threshold
    clean_index = sort_index_loss[:num_use]

    # obtain clean logits and labels
    clean_logits = y_1[clean_index]
    clean_labels = t[clean_index]

    # clean loss calculation
    clean_loss = F.cross_entropy(clean_logits, clean_labels)

    return clean_loss/len(clean_labels)


def cross_entropy_loss(logits, labels):
    # calculate cross entropy loss
    ce_loss = F.cross_entropy(logits, labels)
    return ce_loss/len(labels)


def cross_entropy_loss_update(logits, labels, lowest_loss, indices):
    # calculate cross entropy loss
    ce_loss = F.cross_entropy(logits, labels)

    # update our lowest losses
    lowest_loss.update(indices, F.cross_entropy(
        logits, labels, reduction='none').data.cpu(), logits.data.cpu())

    return ce_loss/len(labels)
