import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import sys

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# Loss functions


def loss_co_ensemble_teaching(y_1, ensemble_y, t):
    # calculate loss for full
    fullLoss = F.cross_entropy(y_1, t)

    # # first find average y_pred for ensemble
    # y_ensemble_avg = torch.mean(ensemble_y, 0)
    # # calculate loss for ensemble
    # ensembleLoss = F.cross_entropy(y_ensemble_avg, t)

    # totalLoss = 0.5 * fullLoss + 0.5 * ensembleLoss

    return fullLoss/len(t)
