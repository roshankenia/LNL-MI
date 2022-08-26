import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.autograd import Variable
import os
import sys
import time
import argparse
from data.cifar import CIFAR10, CIFAR100
from train import train
from utils.featureMap import FeatureMap
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


# function to extract features from data

def extract_features(x_data):
    # define our pretrained resnet
    model = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # remove last fully connected layer from model
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    # input data to model
    data = np.moveaxis(x_data.train_data, -1, 1).float()
    data = torch.from_numpy(data)
    # print(model)
    features = model(data)
    print(features)
    print(features.shape)
