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

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# Loss functions


def combined_relabel(y_1, y_2, indices, combinedLabels, cur_time):

    # calculate combined logits
    combined_logits = (y_1 + y_2)/2

    current_target = combinedLabels.getLabelsOnly(indices)

    # calculate cross-entropy loss using combined logits
    combined_cross_entropy_loss = F.cross_entropy(
        combined_logits, current_target.to(y_1.device), reduction='none')

    # update our labels
    combinedLabels.update(
        combined_logits, combined_cross_entropy_loss, indices, cur_time)

    # obtain clean labels
    useIndices_1, useLabels_1, useActualIndices_1, useIndices_2, useLabels_2, useActualIndices_2 = combinedLabels.getLabels(
        y_1, y_2, combined_cross_entropy_loss, indices)

    # use half labels to update model 1 while other half to update model 2
    loss_1 = F.cross_entropy(y_1[useIndices_1], useLabels_1.to(y_1.device))
    loss_2 = F.cross_entropy(y_2[useIndices_2], useLabels_2.to(y_1.device))

    return loss_1/len(useActualIndices_1), loss_2/len(useActualIndices_2)
