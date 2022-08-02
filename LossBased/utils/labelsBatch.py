import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
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


class BatchLabels():

    def __init__(self, num_samples, train_labels, true_train_labels, noise_or_not, history, num_classes):
        # intialize our data arrays
        self.labels = torch.Tensor([train_labels[i]
                                   for i in range(num_samples)]).long()
        self.predictions = torch.Tensor([torch.zeros(history)
                                         for i in range(num_samples)])
        self.entropy = torch.Tensor([torch.zeros(history)
                                    for i in range(num_samples)])
        self.peak = torch.Tensor([torch.zeros(history)
                                 for i in range(num_samples)])

        self.true_train_labels = [i[0] for i in true_train_labels]
        self.noise_or_not = noise_or_not

        self.history = history
        self.num_classes = num_classes

        self.time = 0

    def addHistory(self, logits, indices):
        # for each sample update its entropy and peak values in our history arrays
        for i in range(len(indices)):
            index = indices[i]
            sampleLogits = logits[i].clone().detach().cpu()
            # obtain prediction
            samplePrediction = torch.argmax(sampleLogits)

            # first take softmax of our logits
            probs = torch.sort(F.softmax(sampleLogits, dim=1)).values
            # calculate entropy
            sampleEntropy = Categorical(probs=probs).entropy()
            # calculate peak value
            probs = torch.flip(probs, dims=(0,))
            samplePeakValue = (probs[0]/probs[1]).item()
            # add our data to the history
            self.predictions[index][self.time] = samplePrediction
            self.entropy[index][self.time] = sampleEntropy
            self.peak[index][self.time] = samplePeakValue

    def addTime(self):
        self.time = self.time + 1

    def reconfirmLabels(self):
        # here we will check the statistics on each sample and see if anything needs to be relabeled
        # first calculate the standard deviation in entropy and peak value for each sample
        entropySTDs = torch.std(self.entropy, dim=1)
        peakSTDs = torch.std(self.peak, dim=1)

        # find average entropy and peak standard deviation
        avgEntStd = torch.mean(entropySTDs)
        avgPeakStd = torch.mean(peakSTDs)

        # now confirm or relabel samples
        lenientRelabelCorrect = 0
        lenientRelabelIncorrect = 0
        lenientReconfirmCorrect = 0
        lenientReconfirmIncorrect = 0
        strictRelabelCorrect = 0
        strictRelabelIncorrect = 0
        strictReconfirmCorrect = 0
        strictReconfirmIncorrect = 0
        for i in range(len(self.labels)):
            # first find highest voted label with number of votes
            votes = torch.zeros(self.num_classes)
            for pred in self.predictions[i]:
                votes[pred] += 1
            voteCounts, mostVotedLabel = torch.max(votes, dim=0)
            if entropySTDs[i] < avgEntStd[i] and peakSTDs[i] < avgPeakStd[i]:
                # can be more lenient with relabeling
                if voteCounts > self.history * 1/2:
                    self.labels[i] = mostVotedLabel
                    if self.true_train_labels[i] == mostVotedLabel:
                        lenientRelabelCorrect += 1
                    else:
                        lenientRelabelIncorrect += 1
                else:
                    if self.true_train_labels[i] == self.labels[i]:
                        lenientReconfirmCorrect += 1
                    else:
                        lenientReconfirmIncorrect += 1
            else:
                # must be more strict with relabeling since jumps a lot
                if voteCounts > self.history * 3/4:
                    self.labels[i] = mostVotedLabel
                    if self.true_train_labels[i] == mostVotedLabel:
                        strictRelabelCorrect += 1
                    else:
                        strictRelabelIncorrect += 1
                else:
                    if self.true_train_labels[i] == self.labels[i]:
                        strictReconfirmCorrect += 1
                    else:
                        strictReconfirmIncorrect += 1

        return lenientRelabelCorrect, lenientRelabelIncorrect, lenientReconfirmCorrect, lenientReconfirmIncorrect, strictRelabelCorrect, strictRelabelIncorrect, strictReconfirmCorrect, strictReconfirmIncorrect

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

    def getLabels(self, indices):
        useLabels = []
        noise_or_not = []
        for i in range(len(indices)):
            index = indices[i]
            # first find label
            label = self.labels[index]
            # check if noisy or not
            if label == self.true_train_labels[index]:
                noise_or_not.append(1)
            else:
                noise_or_not.append(0)
            useLabels.append(label)
        # make labels into tensor
        useLabels = torch.Tensor([useLabels[i]
                                 for i in range(len(useLabels))]).long()
        return useLabels, torch.Tensor(noise_or_not).long()
