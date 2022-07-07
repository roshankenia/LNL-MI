import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from Cifar10Custom import Cifar10BinaryNoisy

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FDModel():
    def __init__(self, x, y, num_epochs=5, batch_size=64, learning_rate=0.001):
        # initialize model with all data and presets for training
        train_dataset = Cifar10BinaryNoisy(x, y)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                        shuffle=True)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = torchvision.models.resnet34(
            pretrained=False, num_classes=1).to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate)

    def train(self):
        # train model
        n_total_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # origin shape: [4, 3, 32, 32] = 4, 3, 1024
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % 20 == 0:
                    print(
                        f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                    # print(binary_acc(outputs, labels))
                    print(outputs)
                    print(labels)

        print('Finished Training')

    def reset(self):
        # to reset the model we simply create a new ResNet
        self.model = torchvision.models.resnet34(
            pretrained=False, num_classes=1).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
