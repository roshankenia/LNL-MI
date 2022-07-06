import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from Cifar10TrainClean import Cifar10BinaryClean
from Cifar10TestClean import Cifar10BinaryCleanTest

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
num_epochs = 100
batch_size = 512
learning_rate = 0.01

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

train_dataset = Cifar10BinaryClean()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_dataset = Cifar10BinaryCleanTest()
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False)

model = torchvision.models.resnet34(pretrained=False, num_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

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
        # scheduler.step()

        if (i+1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            # print(binary_acc(outputs, labels))
            # print(outputs)
            # print(labels)

print('Finished Training')
# PATH = './resnet-mi.pth'
# torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    x = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        y_test_pred = torch.sigmoid(outputs)
        y_pred_tag = torch.round(y_test_pred)

        if x == 0:
            print('labels:', labels[0:20])
            # print('outputs:', outputs[0:20])
            print('predicted:', y_pred_tag[0:20])
            x = 1

        n_correct += (y_pred_tag == labels).sum().float()
        n_samples += len(labels)
    acc = n_correct/n_samples
    acc = torch.round(acc * 100)

    print(f'Accuracy of the network: {acc} %')
