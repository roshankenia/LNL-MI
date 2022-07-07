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
import math

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
        print(x.shape)
        self.y = y
        print(y.shape)
        self.k = k
        self.length = len(y)
        self.interval = int(math.ceil(self.length/k))

    # function to split the dataset into k subsets
    def split(self):
        # create array of indexes
        indexes = list(range(0, self.length))

        # shuffle indexes
        indexes = random.shuffle(indexes)

        # create datasets based on each interval

        # add ranges to data
        x_tensors = []
        y_tensors = []

        index = 0
        while index < self.length:
            # if at the end of the data we do not want to go too far
            range_end = index + self.interval
            if range_end > self.length:
                range_end = self.length
            # add data in range of interval
            x_tensors.append(self.x[index, range_end])
            y_tensors.append(self.y[index, range_end])

            index = index+self.interval

        print(x_tensors)
        print(y_tensors)
        print(len(x_tensors))


x_data = torch.load('cifar10_noisy_data_tensor_nonorm.pt')
y_data = torch.load('cifar10_noisy_ground_truth_tensor_nonorm.pt')
testSplit = KDataSplitter(x_data, y_data, k=4)
testSplit.split()
