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

    # function to split the dataset into k subsets
    def split():
