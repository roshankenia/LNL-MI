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
from sklearn.cluster import KMeans
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
        self.features = torch.zeros(num_samples, num_epochs, num_classes)

        self.history = history
        self.name = name
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.time = 0

    def addTime(self):
        self.time += 1

    def addData(self, logits, indices, epoch):
        # for each sample update its features in our array
        for i in range(len(indices)):
            index = indices[i]
            sampleLogits = logits[i].clone().detach().cpu()

            # first take softmax of our logits
            probs = torch.sort(F.softmax(sampleLogits, dim=0)).values
            self.features[index][self.time] = probs

    def clusterData(self, labels, noise, num_components=2, num_clusters=2):
        # we will run our dimension reduction and then run our clustering method on it

        # first reduce dimensions

        # reshape our features
        usedFeatures = self.features[:, self.time-self.history:self.history, :]
        print(usedFeatures.shape)
        reshapeSize = self.history*self.num_classes
        reshapeFeatures = torch.reshape(
            usedFeatures, (self.num_samples, reshapeSize))

        for label in range(10):
            # find all indexes with this label
            indexes = np.where(np.array(labels) == label)
            print(indexes)
            currentFeatures = reshapeFeatures[indexes]
            print(currentFeatures)
            print(currentFeatures.shape)

            # run tSNE on the current features
            tsne = TSNE(num_components, method='exact')
            tsne_result = tsne.fit_transform(currentFeatures)

            # now run our dimensionality reduction
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(tsne_result)

            # calculate how accurate clustering was
            oneAsCleanCorrect = 0
            oneAsCleanIncorrect = 0
            zeroAsNoisyCorrect = 0
            zeroAsNoisyIncorrect = 0

            oneAsNoisyCorrect = 0
            oneAsNoisyIncorrect = 0
            zeroAsCleanCorrect = 0
            zeroAsCleanIncorrect = 0

            for i in range(len(kmeans.labels_)):
                clusterLabel = kmeans.labels_[i]
                clean = noise[i]
                # when clean = 1 it means it is clean, when clean = 0 it means it is noisy
                if clean == 1:
                    if clusterLabel == 1:
                        oneAsCleanCorrect += 1
                        oneAsNoisyIncorrect += 1
                    elif clusterLabel == 0:
                        zeroAsCleanCorrect += 1
                        zeroAsNoisyIncorrect += 1
                elif clean == 0:
                    if clusterLabel == 1:
                        oneAsNoisyCorrect += 1
                        oneAsCleanIncorrect += 1
                    elif clusterLabel == 0:
                        zeroAsNoisyCorrect += 1
                        zeroAsCleanIncorrect += 1

            oneAsCleanTotal = oneAsCleanCorrect + oneAsCleanIncorrect
            oneAsClean = oneAsCleanCorrect/oneAsCleanTotal
            zeroAsNoisyTotal = zeroAsNoisyCorrect + zeroAsNoisyIncorrect
            zeroAsNoisy = zeroAsNoisyCorrect/zeroAsNoisyTotal
            oneAsNoisyTotal = oneAsNoisyCorrect + oneAsNoisyIncorrect
            oneAsNoisy = oneAsNoisyCorrect/oneAsNoisyTotal
            zeroAsCleanTotal = zeroAsCleanCorrect + zeroAsCleanIncorrect
            zeroAsClean = zeroAsCleanCorrect/zeroAsCleanTotal

            # print accuracies
            print('\tLabel: ', label, ' Number Components: ', num_components)
            print('\t\tOne as clean accuracy: %.4f %%, total correct: %d, total incorrect: %d' %
                  (oneAsClean, oneAsCleanCorrect, oneAsCleanIncorrect))
            print('\t\tZero as noisy accuracy: %.4f %%, total correct: %d, total incorrect: %d' %
                  (zeroAsNoisy, zeroAsNoisyCorrect, zeroAsNoisyIncorrect))
            print('\n\n')
            print('\t\tOne as noisy accuracy: %.4f %%, total correct: %d, total incorrect: %d' %
                  (oneAsNoisy, oneAsNoisyCorrect, oneAsNoisyIncorrect))
            print('\t\tZero as clean accuracy: %.4f %%, total correct: %d, total incorrect: %d' %
                  (zeroAsClean, zeroAsCleanCorrect, zeroAsCleanIncorrect))

    def makePlot(self, epoch, labels, noise):
        print('Making plots')
        # reshape our features
        usedFeatures = self.features[:, self.time-self.history:self.history, :]
        print(usedFeatures.shape)
        reshapeSize = self.history*self.num_classes
        reshapeFeatures = torch.reshape(
            usedFeatures, (self.num_samples, reshapeSize))

        for label in range(10):
            # find all indexes with this label
            indexes = np.where(np.array(labels) == label)
            print(indexes)
            currentFeatures = reshapeFeatures[indexes]
            print(currentFeatures)
            print(currentFeatures.shape)

            # run tSNE on the current features
            n_components = 2
            tsne = TSNE(n_components)
            tsne_result = tsne.fit_transform(currentFeatures)
            tsne_result.shape

            tsne_result_df = pd.DataFrame(
                {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1], 'noise': noise[indexes]})

            # 2D tSNE
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
                            hue='noise', data=tsne_result_df, ax=ax, s=10)
            lim = (tsne_result.min()-5, tsne_result.max()+5)
            plt.title('2D tSNE Sample Predictions over last 10 epochs reduced')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plot_title = 'tSNE-Probs-'+str(label)+'-'+str(epoch+1)+'.png'
            plt.savefig(plot_title)
            plt.close()

            n_components = 3
            tsne = TSNE(n_components)
            tsne_result = tsne.fit_transform(currentFeatures)
            tsne_result.shape

            tsne_result_df = pd.DataFrame(
                {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1], 'tSNE Feature 3': tsne_result[:, 2], 'noise': noise[indexes]})
            # 3D tSNE
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(tsne_result_df['tSNE Feature 1'], tsne_result_df['tSNE Feature 2'],
                       tsne_result_df['tSNE Feature 3'], c=tsne_result_df['noise'])
            lim = (tsne_result.min()-5, tsne_result.max()+5)
            plt.title('3D tSNE Sample Predictions over last 10 epochs reduced')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_zlim(lim)
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plot_title = 'tSNE-Probs-3D-'+str(label)+'-'+str(epoch+1)+'.png'
            plt.savefig(plot_title)
            plt.close()

            # 2D PCA
            # We want to get PCA embedding with 2 dimensions
            n_components = 2
            pca = PCA(n_components)
            pca_result = pca.fit_transform(currentFeatures)
            pca_result.shape
            # Two dimensions for each of our images
            # Plot the result of our PCA with the label color coded
            pca_result_df = pd.DataFrame(
                {'PCA Feature 1': pca_result[:, 0], 'PCA Feature 2': pca_result[:, 1], 'noise': noise[indexes]})
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.scatterplot(x='PCA Feature 1', y='PCA Feature 2',
                            hue='noise', data=pca_result_df, ax=ax, s=10)
            lim = (pca_result.min()-5, pca_result.max()+5)
            plt.title('2D PCA Sample Predictions over last 10 epochs reduced')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plot_title = 'PCA-Probs-'+str(label)+'-'+str(epoch+1)+'.png'
            plt.savefig(plot_title)
            plt.close()

            # 3D PCA
            n_components = 3
            pca = PCA(n_components)
            pca_result = pca.fit_transform(currentFeatures)
            pca_result.shape

            pca_result_df = pd.DataFrame(
                {'PCA Feature 1': pca_result[:, 0], 'PCA Feature 2': pca_result[:, 1], 'PCA Feature 3': pca_result[:, 2], 'noise': noise[indexes]})

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pca_result_df['PCA Feature 1'], pca_result_df['PCA Feature 2'],
                       pca_result_df['PCA Feature 3'], c=pca_result_df['noise'])
            lim = (pca_result.min()-5, pca_result.max()+5)
            plt.title('3D PCA Sample Predictions over last 10 epochs reduced')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_zlim(lim)
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plot_title = 'PCA-Probs-3D-'+str(label)+'-'+str(epoch+1)+'.png'
            plt.savefig(plot_title)
            plt.close()

        # lets also make plots with all classes together
        # run tSNE on the current features
        n_components = 2
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(reshapeFeatures)
        tsne_result.shape
        tsne_result_df = pd.DataFrame(
            {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1], 'noise': noise})

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
                        hue='noise', data=tsne_result_df, ax=ax, s=10)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        plt.title('tSNE for Features')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = 'tSNE-Probs-All-2D-'+str(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()

        # now for 3D
        n_components = 3
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(reshapeFeatures)
        tsne_result.shape

        tsne_result_df = pd.DataFrame(
            {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1], 'tSNE Feature 3': tsne_result[:, 2], 'noise': noise})

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(tsne_result_df['tSNE Feature 1'], tsne_result_df['tSNE Feature 2'],
                   tsne_result_df['tSNE Feature 3'], c=tsne_result_df['noise'])
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        plt.title('tSNE for Features')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = 'tSNE-Probs-All-3D-'+str(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()

        # run PCA on the current features
        n_components = 2
        pca = PCA(n_components)
        pca_result = pca.fit_transform(reshapeFeatures)
        pca_result.shape
        pca_result_df = pd.DataFrame(
            {'PCA Feature 1': pca_result[:, 0], 'PCA Feature 2': pca_result[:, 1], 'noise': noise})

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x='PCA Feature 1', y='PCA Feature 2',
                        hue='noise', data=pca_result_df, ax=ax, s=10)
        lim = (pca_result.min()-5, pca_result.max()+5)
        plt.title('2D PCA for Features')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = 'PCA-Probs-All-2D-'+str(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()

        # now for 3D
        n_components = 3
        pca = TSNE(n_components)
        pca_result = pca.fit_transform(reshapeFeatures)
        pca_result.shape

        pca_result_df = pd.DataFrame(
            {'PCA Feature 1': pca_result[:, 0], 'PCA Feature 2': pca_result[:, 1], 'PCA Feature 3': pca_result[:, 2], 'noise': noise})

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pca_result_df['PCA Feature 1'], pca_result_df['PCA Feature 2'],
                   pca_result_df['PCA Feature 3'], c=pca_result_df['noise'])
        lim = (pca_result.min()-5, pca_result.max()+5)
        plt.title('3D PCA for Features')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = 'PCA-Probs-All-3D-'+str(epoch+1)+'.png'
        plt.savefig(plot_title)
        plt.close()
