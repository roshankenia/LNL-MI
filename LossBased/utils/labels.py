from re import I
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


class LowLossLabels():

    def __init__(self, num_samples, true_train_labels):
        # intialize our data arrays
        self.labels = torch.Tensor([-1 for i in range(num_samples)]).long()
        self.losses = torch.Tensor([-1 for i in range(num_samples)])
        self.true_train_labels = [i[0] for i in true_train_labels]
        print(true_train_labels)

    def update(self, indices, losses, preds, noise_or_not):
        relabelCount = 0
        for i in range(len(indices)):
            index = indices[i]
            loss = losses[i]
            label = torch.argmax(preds[i])
            # initial update
            if self.labels[index] == -1:
                self.labels[index] = label
                self.losses[index] = loss
            else:
                # update labels if losses are smaller
                if loss < self.losses[index]:
                    if self.labels[index] != label:
                        relabelCount += 1
                self.labels[index] = label
                self.losses[index] = loss
        return relabelCount
