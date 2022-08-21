import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class FeatureMap():

    def __init__(self, num_samples, num_epochs, num_classes):
        # intialize our data arrays
        self.features = torch.zeros(num_samples, 10, num_classes)

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_epochs = num_epochs

    def addData(self, logits, indices, epoch):
        # for each sample update its features in our array
        for i in range(len(indices)):
            index = indices[i]
            sampleLogits = logits[i].clone().detach().cpu()

            # first take softmax of our logits
            probs = torch.sort(F.softmax(sampleLogits, dim=0)).values
            time = epoch % 10
            self.features[index][time] = probs

    def makePlot(self, epoch, labels, noise):
        # reshape our features
        reshapeSize = self.num_epochs*self.num_classes
        reshapeFeatures = torch.reshape(
            self.features, (self.num_samples, reshapeSize))
        print(reshapeFeatures.size)
        # We want to get TSNE embedding with 2 dimensions
        n_components = 2
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(reshapeFeatures)
        tsne_result.shape
        # Two dimensions for each of our images
        # Plot the result of our TSNE with the label color coded
        # A lot of the stuff here is about making the plot look pretty and not TSNE
        tsne_result_df = pd.DataFrame(
            {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1], 'label': labels, 'noise': noise})
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
                        hue='label', data=tsne_result_df, ax=ax, s=10)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        plt.title('Sample Predictions over last 10 epochs reduced')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = 'tSNE-Labels-'+(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
                        hue='noise', data=tsne_result_df, ax=ax, s=10)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        plt.title('Sample Predictions over last 10 epochs reduced')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = 'tSNE-Noise-'+(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()
