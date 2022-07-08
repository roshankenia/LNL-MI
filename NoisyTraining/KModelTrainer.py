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
        for i in range(len(x)):
            x_sample = x[i]
            y_sample = y[i]

            # obtain predictions from each model
            predictions = []
            for model in self.models:
                predictions.append(torch.sigmoid(
                    model.predict(x_sample)).item())

            # print(predictions)
            predictions = torch.tensor(predictions)
            # calculate average probability
            y_avg = torch.mean(predictions)
            y_avg = torch.unsqueeze(y_avg, 0)
            # print(y_avg)
            # print(y_sample)

            # compute binary cross entropy loss using this average
            bce = loss(y_avg, y_sample)

            # print('bce:', bce)

            # now we need to compute the furthest apart metric
            distances = []
            for prob_one in predictions:
                for prob_two in predictions:
                    distances.append(np.absolute((prob_one-prob_two)))

            # furthest apart uncertainty is the max of these values
            furthestUncertainty = max(distances)
            # print('furth:', furthestUncertainty)

            bces.append(bce.item())
            furthest.append(furthestUncertainty.item())

            if i % 1000 == 0:
                print(i, 'samples done')

        return bces, furthest
