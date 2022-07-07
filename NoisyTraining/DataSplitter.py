import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from Cifar10TrainNoisy import Cifar10BinaryNoisy
from Cifar10TestNoisy import Cifar10BinaryNoisyTest
import random

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class KDataSplitter():

    def __init__(self, x, y, k):
        # Initialize x, y, and k
        self.x = x
        self.y = y
        self.k = k
        self.length = len(y)
        print('len:', self.length)
        self.interval = int(self.length/k)
        print('interval:', self.interval)

    # function to split the dataset into k subsets
    def split(self):
        # create array of indexes
        indexes = list(range(0, self.length))

        # shuffle indexes
        random.shuffle(indexes)

        # create datasets based on each interval

        # first make empty datasets
        x_arrays = []
        y_arrays = []
        for i in range(self.k):
            x_arrays.append([])
            y_arrays.append([])

        # now add elements based on each interval while using random indexes
        i = 0
        arrayIndex = 0
        while i < self.length:
            # add to appropriate array
            x_arrays[arrayIndex].append(self.x[indexes[i]])
            y_arrays[arrayIndex].append(self.y[indexes[i]].item())
            i += 1

            # if we hit an interval we increase to add to the next array
            if i != 0 and i % self.interval == 0 and i + self.interval <= self.length:
                arrayIndex += 1

        # transform to tensors
        for i in range(len(x_arrays)):
            x_arrays[i] = torch.tensor(torch.stack(
                x_arrays[i]), dtype=torch.float32)
            y_arrays[i] = torch.unsqueeze(torch.tensor(
                y_arrays[i], dtype=torch.float32), 1)

        for arr in x_arrays:
            print(len(arr))
        print()
        for arr in y_arrays:
            print(len(arr))


x_data = torch.load('cifar10_noisy_data_tensor_nonorm.pt')
y_data = torch.load('cifar10_noisy_ground_truth_tensor_nonorm.pt')
testSplit = KDataSplitter(x_data, y_data, k=4)
testSplit.split()
