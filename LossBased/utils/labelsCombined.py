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
        self.labels = torch.Tensor([[train_labels[i]]
                                   for i in range(num_samples)]).long()
        self.losses = torch.Tensor([[3.32]for i in range(num_samples)])
        self.true_train_labels = [i[0] for i in true_train_labels]
        self.noise_or_not = noise_or_not

        self.history = history
        self.num_classes = num_classes

        print(self.labels)
        print(self.losses)

    def update(self, logits, combinedLoss, indices):
        # we will keep track of the indices to use
        for i in range(len(indices)):
            index = indices[i]
            pred = torch.argmax(logits[i])
            loss = combinedLoss[i]

            # obtain current labels and losses
            currentLabels = self.labels[index]
            currentLosses = self.losses[index]

            # if we do not have a long enough history yet just add data to arrays
            if len(currentLabels) < self.history:
                currentLabels.append(pred)
                currentLosses.append(loss)

                self.labels[index] = currentLabels
                self.losses[index] = currentLosses
            else:
                # first sort the losses
                currentLosses, sortIndices = torch.sort(currentLosses)
                currentLabels = currentLabels[sortIndices]

                # check if we have a better loss prediction
                if loss < currentLosses[self.history-1]:
                    currentLosses[self.history-1] = loss
                    currentLabels[self.history-1] = pred

                self.labels[index] = currentLabels
                self.losses[index] = currentLosses

    def getLabelsOnly(self, indices):
        useLabels = []
        for i in range(len(indices)):
            index = indices[i]
            # first find label
            logitsSum = torch.zeros(self.num_classes)
            for j in range(len(self.losses[index])):
                logitsSum[self.labels[index][j]] += self.losses[index][j]
            label = torch.argmax(logitsSum)
            useLabels.append(label)
        # make labels into tensor
        useLabels = torch.Tensor([[useLabels[i]]
                                 for i in range(len(useLabels))])
        return useLabels

    def getLabels(self, y_1, y_2, combinedLoss, indices):
        useIndices_1 = []
        useLabels_1 = []
        useActualIndices_1 = []

        useIndices_2 = []
        useLabels_2 = []
        useActualIndices_2 = []

        combinedLossMean = combinedLoss.mean()
        lowLossCount = 0
        consistentCount = 0
        unusedCount = 0
        count = 0
        for i in range(len(indices)):
            index = indices[i]
            # first find label
            logitsSum = torch.zeros(self.num_classes)
            for j in range(len(self.losses[index])):
                logitsSum[self.labels[index][j]] += self.losses[index][j]
            label = torch.argmax(logitsSum)
            # if a label has a low combined loss we use it
            if combinedLoss[i] < combinedLossMean:
                count += 1
                lowLossCount += 1
                if count % 2 == 1:
                    useLabels_1.append(label)
                    useIndices_1.append(i)
                    useActualIndices_1.append(index)
                else:
                    useLabels_2.append(label)
                    useIndices_2.append(i)
                    useActualIndices_2.append(index)
            # if a label has a high combined loss but is consistent we use it
            elif label == torch.argmax(y_1[i]) and label == torch.argmax(y_2[i]):
                count += 1
                consistentCount += 1
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
                unusedCount += 1

        useLabels_1 = torch.Tensor([[useLabels_1[i]]
                                    for i in range(len(useLabels_1))])
        useLabels_2 = torch.Tensor([[useLabels_2[i]]
                                    for i in range(len(useLabels_2))])
        return useIndices_1, useLabels_1, useActualIndices_1, useIndices_2, useLabels_2, useActualIndices_2
