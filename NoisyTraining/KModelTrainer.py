import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from DataSplitter import KDataSplitter
from ModelTrain import FDModel

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class KModelTrain():

    def __init__(self, x, y, k):
        self.data_splitter = KDataSplitter(x, y, k)
        x_arrays, y_arrays = self.data_splitter.split()

        self.x_arrays = x_arrays
        self.y_arrays = y_arrays
        # set up models
        self.models = []
        for i in range(k):
            # create kth model
            newModel = FDModel(
                self.x_arrays[i], self.y_arrays[i], num_epochs=5, batch_size=64, learning_rate=0.001)
            # train new model
            newModel.train()

            # add to models
            self.models.append(newModel)
