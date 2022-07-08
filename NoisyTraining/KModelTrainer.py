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
            nextIndex = i+1
            if nextIndex == k:
                nextIndex = 0
            x_data = torch.cat(
                (self.x_arrays[i], self.x_arrays[nextIndex]), dim=0)
            y_data = torch.cat(
                (self.y_arrays[i], self.y_arrays[nextIndex]), dim=0)
            # create kth model
            newModel = FDModel(
                x_data, y_data, num_epochs=50, batch_size=64, learning_rate=0.01)
            # train new model
            print('Training model', i)
            newModel.train()

            # add to models
            self.models.append(newModel)

    def calculateUncertainty(self, x, y):
        # for each sample we obtain the prediction probability of it being class 1 from each model
        # we then average these together to feed to binary cross entropy
        # we then compute our uncertainty metric
        loss = nn.BCELoss()
        bces = []
        furthest = []
        modelPredictions = []
        for model in self.models:
            modelPredictions.append(torch.sigmoid(
                model.predict(x)))
        for i in range(len(x)):
            y_sample = y[i]

            # obtain predictions from each model
            predictions = []
            for j in range(len(modelPredictions)):
                predictions.append(modelPredictions[j][i].item())
            predictions = torch.tensor(predictions)
            # calculate average probability
            y_avg = torch.mean(predictions)
            y_avg = torch.unsqueeze(y_avg, 0)

            # print(y_avg)
            # print(y_sample)

            # compute binary cross entropy loss using this average
            bce = loss(y_avg, y_sample)

            # print('bce:', bce)

            # furthest apart uncertainty is the difference between the maximum prediction and the minimum prediction
            furthestUncertainty = max(predictions) - min(predictions)
            # print('furth:', furthestUncertainty, res)

            bces.append(bce.item())
            furthest.append(furthestUncertainty.item())

            # if i % 1000 == 0:
            #     print(i, 'samples done')

        return bces, furthest
