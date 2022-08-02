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


def loss_coteaching(y_1, y_2, t, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce=False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    # find number of samples to use
    num_use_1 = torch.nonzero(loss_1 < loss_1.mean()).shape[0]
    num_use_2 = torch.nonzero(loss_2 < loss_2.mean()).shape[0]

    pure_ratio_1 = np.sum(
        noise_or_not[ind_1_sorted[:num_use_1]].numpy())/float(num_use_1)
    pure_ratio_2 = np.sum(
        noise_or_not[ind_2_sorted[:num_use_2]].numpy())/float(num_use_2)

    ind_1_update = ind_1_sorted[:num_use_1]
    ind_2_update = ind_2_sorted[:num_use_2]
    # exchange
    loss_1_update = F.cross_entropy(
        y_1[ind_2_update], t[ind_2_update])
    # print('before:', loss_1_update)
    # print('after:', torch.sum(loss_1_update)/num_remember)
    loss_2_update = F.cross_entropy(
        y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_use_1, torch.sum(loss_2_update)/num_use_2, pure_ratio_1, pure_ratio_2
