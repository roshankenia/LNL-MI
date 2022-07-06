import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import sys
import torchvision.transforms as transforms


# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class Cifar10BinaryClean(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # here the first column is the class label, the rest are the features
        # size [n_samples, n_features]
        self.x_data = torch.load('../../cifar10_clean_data_tensor.pt')
        print('x shape:', self.x_data.shape)
        self.y_data = torch.load(
            '../../cifar10_clean_ground_truth_tensor.pt')  # size [n_samples, 1]
        print('y shape:', self.y_data.shape)
        self.n_samples = self.x_data.shape[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# # create dataset
# dataset = Cifar10BinaryClean()

# # get first sample and unpack
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
