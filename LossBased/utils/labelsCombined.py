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


class EpochLabels():

    def __init__(self):
        # intialize our data arrays
        self.preds = None

    def update(self, preds):
        # initial update
        if self.preds is None:
            self.preds = preds
        else:
            # update preds in ratio
            for i in range(len(preds)):
                self.preds[i] = 0.75 * self.preds[i] + 0.25 * preds[i]

        return self.preds
