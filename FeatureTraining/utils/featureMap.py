import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
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


class FeatureMap():

    def __init__(self, num_samples, num_epochs, num_classes):
        # intialize our data arrays
        self.features = torch.zeros(num_samples, num_epochs, num_classes)

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_epochs = num_epochs

    def addData(self, logits, indices, epoch):
        # for each sample update its features in our array
        for i in range(len(indices)):
            index = indices[i]
            sampleLogits = logits[i].clone().detach().cpu()

            # first take softmax of our logits
            probs = torch.sort(F.softmax(sampleLogits, dim=0)).values
            self.features[index][epoch] = probs
