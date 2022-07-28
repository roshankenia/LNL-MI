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


def combined_relabel(y_1, y_2, t, indices, combinedLabels, cur_time):

    # calculate combined logits
    combined_logits = (y_1 + y_2)/2
    print(combined_logits)
    # begin = time.time()
    current_target = combinedLabels.getLabelsOnly(indices)
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
    # end = time.time()
    # print('clean labels:', (end-begin))
    # use half labels to update model 1 while other half to update model 2
    loss_1 = F.cross_entropy(y_1[useIndices_1], useLabels_1.to(t.device))
    loss_2 = F.cross_entropy(y_2[useIndices_2], useLabels_2.to(t.device))

    return loss_1/len(useActualIndices_1), loss_2/len(useActualIndices_2), lowLossCount, consistentCount, unusedCount, lowLossClean, consistentClean, unusedClean
