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


class CombinedLabels():

    def __init__(self, num_samples, train_labels, true_train_labels, noise_or_not):
        # intialize our data arrays
        self.labels = torch.Tensor([[train_labels[i]]
                                   for i in range(num_samples)]).long()
        self.losses = torch.Tensor([[3.32]for i in range(num_samples)])
        self.true_train_labels = [i[0] for i in true_train_labels]
        self.noise_or_not = noise_or_not

        print(self.labels)
        print(self.losses)

    def update(self, preds):
        # initial update
        if len(self.preds) == 0:
            self.preds = preds
        else:
            # update preds in ratio
            for i in range(len(preds)):
                self.preds[i] = 0.75 * self.preds[i] + 0.25 * preds[i]

        return self.preds
