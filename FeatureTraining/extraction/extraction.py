import torch
import torchvision
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


# function to extract features from data

def extract_features(x_data):
    # define our pretrained resnet
    model = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # remove last fully connected layer from model
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    # input data to model
    data = np.moveaxis(x_data.train_data, -1, 1)
    data = torch.from_numpy(data).float()
    # print(model)
    features = model(data)
    features = torch.squeeze(features)

    return features


def make_plots(features, labels, noise_or_not, num_classes):
    for label in range(num_classes):
        # find all indexes with this label
        indexes = np.where(np.array(labels) == label)
        print(indexes)
        currentFeatures = features[indexes].detach().numpy()
        print(currentFeatures)
        print(currentFeatures.shape)

        # run tSNE on the current features
        n_components = 3
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(currentFeatures)
        tsne_result.shape

        tsne_result_df = pd.DataFrame(
            {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1], 'tSNE Feature 3': tsne_result[:, 2], 'noise': noise_or_not[indexes]})

        # fig, ax = plt.subplots(figsize=(10, 10))
        # sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
        #                 hue='noise', data=tsne_result_df, ax=ax, s=10)
        # lim = (tsne_result.min()-5, tsne_result.max()+5)
        # plt.title('tSNE for Features')
        # ax.set_xlim(lim)
        # ax.set_ylim(lim)
        # ax.set_aspect('equal')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # plot_title = 'tSNE-Features-'+str(label)+'.png'
        # plt.savefig(plot_title)
        # plt.close()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(tsne_result_df['tSNE Feature 1'], tsne_result_df['tSNE Feature 2'],
                   tsne_result_df['tSNE Feature 3'], c=tsne_result_df['noise'])
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        plt.title('tSNE for Features')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plot_title = 'tSNE-Features-3D-'+str(label)+'.png'
        plt.savefig(plot_title)
        plt.close()
