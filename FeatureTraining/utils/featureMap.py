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
from sklearn.decomposition import PCA
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

    def __init__(self, num_samples, num_epochs, num_classes, name, history=10):
        # intialize our data arrays
        self.features = torch.zeros(num_samples, history, num_classes)

        self.history = history
        self.name = name
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
        print('Making plots')
        # reshape our features
        reshapeSize = self.history*self.num_classes
        reshapeFeatures = torch.reshape(
            self.features, (self.num_samples, reshapeSize))
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
        plt.title('tSNE Sample Predictions over last 10 epochs reduced')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = self.name+'-tSNE-Labels-'+str(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
                        hue='noise', data=tsne_result_df, ax=ax, s=10)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        plt.title('tSNE Sample Predictions over last 10 epochs reduced')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = self.name+'-tSNE-Noise-'+str(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()

        # # We want to get PCA embedding with 2 dimensions
        # n_components = 2
        # pca = PCA(n_components)
        # pca_result = pca.fit_transform(reshapeFeatures)
        # pca_result.shape
        # # Two dimensions for each of our images
        # # Plot the result of our TSNE with the label color coded
        # # A lot of the stuff here is about making the plot look pretty and not TSNE
        # pca_result_df = pd.DataFrame(
        #     {'PCA Feature 1': pca_result[:, 0], 'PCA Feature 2': pca_result[:, 1], 'label': labels, 'noise': noise})
        # fig, ax = plt.subplots(figsize=(10, 10))
        # sns.scatterplot(x='PCA Feature 1', y='PCA Feature 2',
        #                 hue='label', data=pca_result_df, ax=ax, s=10)
        # lim = (pca_result.min()-5, pca_result.max()+5)
        # plt.title('PCA Sample Predictions over last 10 epochs reduced')
        # ax.set_xlim(lim)
        # ax.set_ylim(lim)
        # ax.set_aspect('equal')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # plot_title = 'pca-Labels-'+str(epoch+1)+'.png'
        # plt.savefig(plot_title)
        # plt.close()

        # fig, ax = plt.subplots(figsize=(10, 10))
        # sns.scatterplot(x='PCA Feature 1', y='PCA Feature 2',
        #                 hue='noise', data=pca_result_df, ax=ax, s=10)
        # lim = (pca_result.min()-5, pca_result.max()+5)
        # plt.title('PCA Sample Predictions over last 10 epochs reduced')
        # ax.set_xlim(lim)
        # ax.set_ylim(lim)
        # ax.set_aspect('equal')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # plot_title = 'pca-Noise-'+str(epoch+1)+'.png'
        # plt.savefig(plot_title)
        # plt.close()
