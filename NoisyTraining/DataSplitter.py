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
        indexes = random.shuffle(indexes)

        # create datasets based on each interval

        # first make empty datasets
        x_arrays = []
        y_arrays = []
        for i in range(self.k):
            x_arrays.append([])
            y_arrays.append([])

        # now add elements based on each interval while using random indexes
        i = 0
        while i < self.length:
            arrayIndex = i % self.interval
            x_arrays[arrayIndex].append(self.x[indexes[i]])
            y_arrays[arrayIndex].append(self.y[indexes[i]])

        print(x_arrays)
        print(y_arrays)
        print(len(x_arrays))


x_data = torch.load('cifar10_noisy_data_tensor_nonorm.pt')
y_data = torch.load('cifar10_noisy_ground_truth_tensor_nonorm.pt')
testSplit = KDataSplitter(x_data, y_data, k=4)
testSplit.split()
