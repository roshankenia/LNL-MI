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


class CombinedLabels():

    def __init__(self, num_samples, train_labels, true_train_labels, noise_or_not, history, num_classes):
        # intialize our data arrays
        self.labels = torch.Tensor([[train_labels[i], -1, -1, -1, -1, -1, -1, -1, -1, -1]
                                   for i in range(num_samples)]).long()
        self.losses = torch.Tensor([[3.32, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                                   for i in range(num_samples)])
        self.counts = torch.zeros(num_samples, num_classes)

        # update first label
        for i in range(num_samples):
            self.counts[i][train_labels[i]] += 1
        self.true_train_labels = [i[0] for i in true_train_labels]
        self.noise_or_not = noise_or_not

        self.history = history
        self.num_classes = num_classes

    def update(self, logits, combinedLoss, indices, cur_time):
        # we will keep track of the indices to use
        for i in range(len(indices)):
            index = indices[i]
            pred = torch.argmax(logits[i])
            loss = combinedLoss[i]

            # if we do not have a long enough history yet just add data to arrays
            if cur_time < self.history:
                self.counts[index][pred] += 1
                self.labels[index][cur_time] = pred
                self.losses[index][cur_time] = loss

                # print(self.counts[index],
                #       self.labels[index], self.losses[index])
            else:
                # first find max loss we have
                # print(self.losses[index][0])
                maxLoss, maxIndex = torch.max(self.losses[index], dim=0)

                # check if we have a better loss prediction
                if loss < maxLoss:
                    # remove max count from counts
                    self.counts[index][self.labels[index][maxIndex]] -= 1
                    # add new count
                    self.counts[index][pred] += 1
                    # set new data
                    self.labels[index][maxIndex] = pred
                    self.losses[index][maxIndex] = loss

    def getBasedOnCount(self, indices):
        useIndices_1 = []
        useLabels_1 = []
        useActualIndices_1 = []

        useIndices_2 = []
        useLabels_2 = []
        useActualIndices_2 = []

        lowLossCount = 0
        lowLossClean = 0
        consistentCount = 0
        consistentClean = 0
        unusedCount = 0
        unusedClean = 0

        count = 0
        for i in range(len(indices)):
            index = indices[i]
            # first find label
            maxLabel, label = torch.max(self.counts[index], dim=0)
            if i == 64:
                print(self.counts[index], maxLabel)
            # only use label if it has majority
            if maxLabel > 9:
                count += 1
                lowLossCount += 1
                if label == self.true_train_labels[index]:
                    lowLossClean += 1
                if count % 2 == 1:
                    useLabels_1.append(label)
                    useIndices_1.append(i)
                    useActualIndices_1.append(index)
                else:
                    useLabels_2.append(label)
                    useIndices_2.append(i)
                    useActualIndices_2.append(index)
            else:
                if label == self.true_train_labels[index]:
                    unusedClean += 1
                unusedCount += 1
        # make labels into tensor
        useLabels_1 = torch.Tensor([useLabels_1[i]
                                    for i in range(len(useLabels_1))]).long()
        useLabels_2 = torch.Tensor([useLabels_2[i]
                                    for i in range(len(useLabels_2))]).long()
        return useIndices_1, useLabels_1, useActualIndices_1, useIndices_2, useLabels_2, useActualIndices_2, lowLossCount, consistentCount, unusedCount, lowLossClean, consistentClean, unusedClean

    def getLabelsOnly(self, indices):
        useLabels = []
        for i in range(len(indices)):
            index = indices[i]
            # first find label
            label = torch.argmax(self.counts[index])
            useLabels.append(label)
        # make labels into tensor
        useLabels = torch.Tensor([useLabels[i]
                                 for i in range(len(useLabels))]).long()
        return useLabels

    def getLabels(self, y_1, y_2, combinedLoss, indices):
        useIndices_1 = []
        useLabels_1 = []
        useActualIndices_1 = []

        useIndices_2 = []
        useLabels_2 = []
        useActualIndices_2 = []

        combinedLossMean = 0.5*combinedLoss.mean()
        lowLossCount = 0
        lowLossClean = 0
        consistentCount = 0
        consistentClean = 0
        unusedCount = 0
        unusedClean = 0
        count = 0
        for i in range(len(indices)):
            index = indices[i]
            # first find label
            label = torch.argmax(self.counts[index])
            # if a label has a low combined loss we use it
            if combinedLoss[i] < combinedLossMean:
                count += 1
                lowLossCount += 1
                if label == self.true_train_labels[index]:
                    lowLossClean += 1
                if count % 2 == 1:
                    useLabels_1.append(label)
                    useIndices_1.append(i)
                    useActualIndices_1.append(index)
                else:
                    useLabels_2.append(label)
                    useIndices_2.append(i)
                    useActualIndices_2.append(index)
            # if a label has a high combined loss but is consistent we use it
            elif label == torch.argmax(y_1[i]) or label == torch.argmax(y_2[i]):
                count += 1
                consistentCount += 1
                if label == self.true_train_labels[index]:
                    consistentClean += 1
                if count % 2 == 1:
                    useLabels_1.append(label)
                    useIndices_1.append(i)
                    useActualIndices_1.append(index)
                else:
                    useLabels_2.append(label)
                    useIndices_2.append(i)
                    useActualIndices_2.append(index)
            # if a label has a high combined loss and is inconsistent we don't use it
            else:
                if label == self.true_train_labels[index]:
                    unusedClean += 1
                unusedCount += 1

        useLabels_1 = torch.Tensor([useLabels_1[i]
                                    for i in range(len(useLabels_1))]).long()
        useLabels_2 = torch.Tensor([useLabels_2[i]
                                    for i in range(len(useLabels_2))]).long()

        return useIndices_1, useLabels_1, useActualIndices_1, useIndices_2, useLabels_2, useActualIndices_2, lowLossCount, consistentCount, unusedCount, lowLossClean, consistentClean, unusedClean
