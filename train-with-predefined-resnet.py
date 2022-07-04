import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from ResNet34 import ResNet34
from MIDatasetTrain import MedicalData
from MIDatasetTest import MedicalDataTest

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

# Hyper-parameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                              download=True, transform=transform)

# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                             download=True, transform=transform)
train_dataset = MedicalData()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_dataset = MedicalDataTest()
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = torchvision.models.resnet34(pretrained=False, num_classes=1)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            # print(binary_acc(outputs, labels))
            print(outputs)
            print(labels)

print('Finished Training')
# PATH = './resnet-mi.pth'
# torch.save(model.state_dict(), PATH)

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     n_class_correct = [0 for i in range(10)]
#     n_class_samples = [0 for i in range(10)]
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs, 1)
#         # print('labels:', labels)
#         # print('outputs:', outputs)
#         # print('predicted:', predicted)

#         y_test_pred = torch.sigmoid(outputs)
#         y_pred_tag = torch.round(y_test_pred)

#         # print('y_pred_tag:', y_pred_tag)
#     #     n_samples += labels.size(0)
#     #     n_correct += (predicted == labels).sum().item()
#     #     for i in range(len(labels)):
#     #         label = labels[i]
#     #         pred = predicted[i]
#     #         if (label == pred):
#     #             n_class_correct[label] += 1
#     #         n_class_samples[label] += 1

#     # acc = 100.0 * n_correct / n_samples
#     # print(f'Accuracy of the network: {acc} %')

#     # for i in range(10):
#     #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
#     #     print(f'Accuracy of {classes[i]}: {acc} %')
