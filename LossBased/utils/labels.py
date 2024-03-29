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

    def __init__(self, num_samples, true_train_labels, noise_or_not):
        # intialize our data arrays
        self.labels = torch.Tensor([-1 for i in range(num_samples)]).long()
        self.losses = torch.Tensor([-1 for i in range(num_samples)])
        self.true_train_labels = [i[0] for i in true_train_labels]
        self.noise_or_not = noise_or_not

    def update(self, indices, losses, preds, epoch):
        # threshold of how much of previous loss is needed to be relabeled
        threshold = 0.5
        if epoch > 40 and epoch < 75:
            threshold = 0.25
        elif epoch >= 75 and epoch < 100:
            threshold = 0.1
        elif epoch >= 100:
            threshold = 0.05

        relabelCount = 0
        correctRelabelCount = 0
        incorrectRelabelCount = 0
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
                if loss < threshold * self.losses[index]:
                    # check if relabeling
                    if self.labels[index] != label:
                        relabelCount += 1
                        # check if relabel will be correct or not
                        if label == self.true_train_labels[index]:
                            correctRelabelCount += 1
                            self.noise_or_not[index] = 1
                        else:
                            incorrectRelabelCount += 1
                            self.noise_or_not[index] = 0
                self.labels[index] = label
                self.losses[index] = loss
        return (relabelCount, correctRelabelCount, incorrectRelabelCount)
