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

    def __init__(self):
        # intialize our data arrays
        self.preds = []
        self.losses = []

    def update(self, losses, preds):
        # initial update
        if len(self.preds) == 0:
            self.preds = preds
            self.losses = losses
        else:
            # update preds if losses are smaller
            for i in range(len(preds)):
                if losses[i] < self.losses[i]:
                    self.preds[i] = preds[i]
                    self.losses[i] = losses[i]

        return self.preds
